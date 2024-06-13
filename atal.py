import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from vision_transformer import ATALHead, ATALHeadClassify

import wandb
import ipdb

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] + torchvision_archs,
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.99, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""") # MOMENTUM
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--image_iterations', default=10, type=int, help="""Number of times image 
        should be trained on in an epoch.""") # ITERATIONS PER IMAGE
    parser.add_argument('--batch_size_per_gpu', default=1, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.') # BATCH SIZE PER GPU
    parser.add_argument('--epochs', default=400, type=int, help='Number of epochs of training.') # EPOCHS
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=5e-4, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""") # LEARNING RATE
    parser.add_argument("--warmup_epochs", default=40, type=int,
        help="Number of epochs for the linear learning-rate warm up.") # WARMUP
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""") # END LEARNING RATE
    parser.add_argument('--min_atal_loss', default='4.0', type=str, help="""Minimum loss 
        required to finish training epoch""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--global_crops_number', type=int, default=2, help="""Number of large
         views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/home/rushil/data/clear-10-train/labeled_images/1', type=str, 
                        help='Please specify path to the training data.')
    parser.add_argument('--val_path', default='/home/rushil/data/clear-10-test/labeled_images/1', type=str, 
                        help='Please specify path to the validation data.')
    parser.add_argument('--postprocess_path', default='/home/rushil/data/clear-10-test/labeled_images/', type=str, 
                        help='Please specify path to the validation data.')
    parser.add_argument('--output_dir', default="/home/rushil/atal/outputs/output_atal", type=str, 
                        help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=10, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--print_freq', default=1, type=int, help='Print model stats every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--num_labels', default=100, type=float,  help="""Number of target labels in dataset.""")
    parser.add_argument('--postprocess_eval', default=False, type=bool,  help="""Set DINO to postprocess eval mode.""")
    parser.add_argument('--postprocess_auto', default=False, type=bool,  help="""Postprocess DINO after training 
                        automatically.""")
    parser.add_argument('--postprocess_train', default=False, type=bool,  help="""Train a seperate linear classifier 
                        in the postprocessing.""")
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up 
                        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


def train_dino(args):

    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    wandb.init(
        project="DINO-Version-2",
        group="CIFAR100-Version-16",
        job_type="Train",
        id=f"cifar100v16-worker-{args.rank:03}",
        config={"architecture": "DINO", "dataset": "CIFAR-100", "epochs": args.epochs,},
        resume=True,
    )

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )

    # train_dataset = datasets.ImageFolder(args.data_path, transform=transform)
    train_dataset = datasets.CIFAR100(root='/home/rushil/data', train=True, download=True, transform=transform)
    # val_dataset = datasets.ImageFolder(args.val_path, transform=transform)
    val_dataset = datasets.CIFAR100(root='/home/rushil/data', train=False, download=True, transform=transform)

    # dataset = datasets.ImageNet(root='/home/rushil/data', split='train', transform=transform)
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_sampler = torch.utils.data.DistributedSampler(val_dataset, shuffle=True)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(train_dataset)} images.")
    print(f"Validation data loaded: there are {len(val_dataset)} images.")

    # ============ building student and teacher networks ... ============

    args.arch = args.arch.replace("deit", "vit")

    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim

    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim

    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    student = utils.MultiCropWrapper(student, ATALHead(embed_dim, args.out_dim, use_bn=args.use_bn_in_head, 
                                                       norm_last_layer=args.norm_last_layer))

    teacher = utils.MultiCropWrapper(teacher, ATALHead(embed_dim, args.out_dim, args.use_bn_in_head))

    student, teacher = student.cuda(), teacher.cuda()

    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], find_unused_parameters=True)
        teacher_without_ddp = teacher.module
    else:
        teacher_without_ddp = teacher
    
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=True)
    teacher_without_ddp.load_state_dict(student.module.state_dict())

    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    linear_student = LinearClassifier(args.out_dim, args.num_labels)
    linear_student = linear_student.cuda()
    linear_student = nn.parallel.DistributedDataParallel(linear_student, device_ids=[args.gpu])

    linear_vit_student = LinearClassifier(student.module.backbone.embed_dim, args.num_labels)
    linear_vit_student = linear_vit_student.cuda()
    linear_vit_student = nn.parallel.DistributedDataParallel(linear_vit_student, device_ids=[args.gpu])

    # ============ preparing loss ... ============

    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + args.global_crops_number, 
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    linear_student_optimizer = torch.optim.SGD(linear_student.parameters(), lr=1e-2, momentum=0.99, weight_decay=1e-5)
    linear_student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(linear_student_optimizer, 
        T_max=args.epochs * len(data_loader), eta_min=1e-5)
    # args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.0, 
    # linear_student_scheduler.step()

    linear_vit_student_optimizer = torch.optim.SGD(linear_vit_student.parameters(), lr=1e-2, momentum=0.99, weight_decay=1e-5)
    linear_vit_student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(linear_vit_student_optimizer, 
        T_max=args.epochs * len(data_loader), eta_min=1e-5)
    # linear_vit_student_scheduler.step()

    # ============ init schedulers ... ============

    iteration_schedule = lambda x: max(int(args.image_iterations * np.exp(-x / (args.epochs / 2))), 1)
    # iteration_schedule = lambda x: args.image_iterations

    lr_schedule = utils.cosine_scheduler(args.lr, args.min_lr, args.epochs, len(data_loader), 
                                         warmup_epochs=args.warmup_epochs, sch=iteration_schedule)

    wd_schedule = utils.cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs, len(data_loader), 
                                         sch=iteration_schedule)

    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(data_loader), 
                                               sch=iteration_schedule)
    
    print(f"Loss, Optimizer and Schedulers ready.")

    # ============ optionally resume training ... ============

    to_restore = {"epoch": 0}

    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
        student_linear_classifier=linear_student,
        vit_student_linear_classifier=linear_vit_student,
        student_linear_classifier_optimizer=linear_student_optimizer,
        vit_student_linear_classifier_optimizer=linear_vit_student_optimizer,
        student_linear_classifier_scheduler=linear_vit_student_scheduler,
        vit_student_linear_classifier_scheduler=linear_vit_student_scheduler,
    )
    
    # utils.restart_from_checkpoint(
    #     os.path.join(args.output_dir, "dino_deitsmall16_pretrain_full_checkpoint.pth"),
    #     run_variables=to_restore,
    #     student=student,
    #     teacher=teacher,
    #     optimizer=optimizer,
    #     fp16_scaler=fp16_scaler,
    #     dino_loss=dino_loss,
    # )

    start_epoch = to_restore["epoch"]

    if args.postprocess_eval:
        print("Starting DINO Postprocess Evaluation !!!")
        postprocess(student, linear_student, linear_vit_student, args)
    else:
        start_time = time.time()
        print("Starting DINO training !!!")
        for epoch in range(start_epoch, args.epochs):
            data_loader.sampler.set_epoch(epoch)

            # ============ training one epoch of DINO ... ============
            train_stats = train_one_epoch(student, teacher, linear_student, linear_vit_student, linear_student_optimizer, 
                                          linear_vit_student_optimizer, teacher_without_ddp, dino_loss, data_loader, optimizer, 
                                          lr_schedule, wd_schedule, momentum_schedule, iteration_schedule, epoch, fp16_scaler, args)
            
            test_one_epoch(student, linear_student, linear_vit_student, val_data_loader, epoch, args)
            
            linear_student_scheduler.step()
            linear_vit_student_scheduler.step()

            # ============ writing logs ... ============
            save_dict = {
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'args': args,
                'dino_loss': dino_loss.state_dict(),
                'student_linear_classifier': linear_student.state_dict(),
                'vit_student_linear_classifier': linear_vit_student.state_dict(),
                'student_linear_classifier_optimizer': linear_student_optimizer.state_dict(),
                'vit_student_linear_classifier_optimizer': linear_vit_student_optimizer.state_dict(),
                'student_linear_classifier_scheduler': linear_student_scheduler.state_dict(),
                'vit_student_linear_classifier_scheduler': linear_vit_student_scheduler.state_dict(),
            }
            if fp16_scaler is not None:
                save_dict['fp16_scaler'] = fp16_scaler.state_dict()
            utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
            if args.saveckp_freq and epoch % args.saveckp_freq == 0:
                utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch}.pth'))
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch}
            if utils.is_main_process():
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        if args.postprocess_auto:
            postprocess(student, linear_student, linear_vit_student, args)

        wandb.join()
        wandb.finish()


def train_one_epoch(student, teacher, linear_student, linear_vit_student, linear_student_opt, linear_vit_student_opt,
                    teacher_without_ddp, dino_loss, data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule, 
                    iteration_schedule, epoch, fp16_scaler, args):
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}] --> '.format(epoch, args.epochs)
    iteration_range = iteration_schedule(epoch)

    student_acc_count = 0
    vit_student_acc_count = 0
    it_count = 0

    for it, (images, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):

        for _ in range(iteration_range):

            it = len(data_loader) * epoch + it

            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                if i == 0:
                    param_group["weight_decay"] = wd_schedule[it]

            images = [im.cuda(non_blocking=True) for im in images]

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_output, student_input = student(images)
                teacher_output, _ = teacher(images[:2])
                loss = dino_loss(student_output, teacher_output, epoch)

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), force=True)
                sys.exit(1)

            optimizer.zero_grad()
            param_norms = None

            if fp16_scaler is None:
                loss.backward()
                if args.clip_grad:
                    param_norms = utils.clip_gradients(student, args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                if args.clip_grad:
                    fp16_scaler.unscale_(optimizer)
                    param_norms = utils.clip_gradients(student, args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            with torch.no_grad():
                m = momentum_schedule[it]
                for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # if loss <= float(args.min_atal_loss):
            #     break

            wandb.log({"aaa) total_loss": loss.item()})

        target = target.cuda()
        target_size = target.size(0)
        it_count = it_count + target_size

        linear_student.train()
        linear_vit_student.train()

        student_detached = student_output.detach()
        student_detached = student_detached[:target_size, :]
        student_input = student_input[:target_size, :]

        linear_student_output = linear_student(student_detached)
        linear_vit_student_output = linear_vit_student(student_input)

        linear_student_loss_result = nn.CrossEntropyLoss()(linear_student_output, target)
        linear_vit_student_loss_result = nn.CrossEntropyLoss()(linear_vit_student_output, target)

        linear_student_opt.zero_grad()
        linear_student_loss_result.backward()

        linear_vit_student_opt.zero_grad()
        linear_vit_student_loss_result.backward()

        linear_student_opt.step()
        linear_vit_student_opt.step()

        linear_student_max = torch.argmax(linear_student_output, dim=-1)
        linear_student_count = torch.sum(linear_student_max == target).item()

        linear_vit_student_max = torch.argmax(linear_vit_student_output, dim=-1)
        linear_vit_student_count = torch.sum(linear_vit_student_max == target).item()

        student_acc_count = student_acc_count + linear_student_count
        vit_student_acc_count = vit_student_acc_count + linear_vit_student_count

        linear_student_count = float(linear_student_count / target_size) * 100
        linear_vit_student_count = float(linear_vit_student_count / target_size) * 100

        student_acc1, student_acc5 = utils.accuracy(linear_student_output, target, topk=(1,5))
        vit_student_acc1, vit_student_acc5 = utils.accuracy(linear_vit_student_output, target, topk=(1,5))

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(s_model_loss=linear_student_loss_result.item())
        metric_logger.update(s_model_count=linear_student_count)
        metric_logger.update(s_model_acc1=student_acc1.item())
        metric_logger.update(s_model_acc5=student_acc5.item())
        metric_logger.update(s_vit_loss=linear_vit_student_loss_result.item())
        metric_logger.update(s_vit_count=linear_vit_student_count)
        metric_logger.update(s_vit_acc1=vit_student_acc1.item())
        metric_logger.update(s_vit_acc5=vit_student_acc5.item())

        wandb.log({"aaa) total_loss": loss.item(), 
                   "aab) lr": optimizer.param_groups[0]["lr"], 
                   "aac) wd": optimizer.param_groups[0]["weight_decay"], 
                   "ada) s_model_loss": linear_student_loss_result.item(),
                   "adb) s_model_lr": linear_student_opt.param_groups[0]["lr"], 
                   "adc) s_model_acc": linear_student_count,
                   "aea) s_vit_loss": linear_vit_student_loss_result.item(),
                   "aeb) s_vit_lr": linear_vit_student_opt.param_groups[0]["lr"], 
                   "aec) s_vit_acc": linear_vit_student_count,  
                   })

    student_epoch_acc = float(student_acc_count / it_count) * 100
    vit_student_epoch_acc = float(vit_student_acc_count / it_count) * 100

    wandb.log({"aad) total_global_loss": metric_logger.loss.global_avg, 
               "aba) s_model_global_loss": metric_logger.s_model_loss.global_avg,
               "abb) s_model_global_acc": student_epoch_acc,
               "abc) s_model_global_acc1": metric_logger.s_model_acc1.global_avg, 
               "abd) s_model_global_acc5": metric_logger.s_model_acc5.global_avg, 
               "aca) s_vit_global_loss": metric_logger.s_vit_loss.global_avg,
               "acb) s_vit_global_acc": vit_student_epoch_acc, 
               "acc) s_vit_global_acc1": metric_logger.s_vit_acc1.global_avg, 
               "acd) s_vit_global_acc5": metric_logger.s_vit_acc5.global_avg, 
               })

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(student, linear_student, linear_vit_student, val_loader, epoch, args):
    
    student_acc_count = 0
    vit_student_acc_count = 0
    it_count = 0

    linear_student.eval()
    linear_vit_student.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Test Epoch [{epoch}/{args.epochs}]'

    for inp, target in metric_logger.log_every(val_loader, args.print_freq, header):

        inp = [im.cuda(non_blocking=True) for im in inp]
        target = target.cuda(non_blocking=True)

        target_size = target.size(0)

        with torch.no_grad():
            student_output, student_input = student(inp)

        student_output = student_output[:target_size, :]
        student_input = student_input[:target_size, :]

        linear_student_output = linear_student(student_output)
        linear_vit_student_output = linear_vit_student(student_input)

        linear_student_loss_result = nn.CrossEntropyLoss()(linear_student_output, target)
        linear_vit_student_loss_result = nn.CrossEntropyLoss()(linear_vit_student_output, target)

        linear_student_max = torch.argmax(linear_student_output, dim=-1)
        linear_student_count = torch.sum(linear_student_max == target).item()

        linear_vit_student_max = torch.argmax(linear_vit_student_output, dim=-1)
        linear_vit_student_count = torch.sum(linear_vit_student_max == target).item()

        student_acc_count = student_acc_count + linear_student_count
        vit_student_acc_count = vit_student_acc_count + linear_vit_student_count
        it_count = it_count + target_size

        linear_student_count = float(linear_student_count / target_size) * 100
        linear_vit_student_count = float(linear_vit_student_count / target_size) * 100


        student_acc1, student_acc5 = utils.accuracy(linear_student_output, target, topk=(1, 3))
        vit_student_acc1, vit_student_acc5 = utils.accuracy(linear_vit_student_output, target, topk=(1, 3))

        metric_logger.update(s_model_loss=linear_student_loss_result.item())
        metric_logger.update(s_vit_loss=linear_vit_student_loss_result.item())

        metric_logger.meters['s_model_acc1'].update(student_acc1.item(), n=args.batch_size_per_gpu)
        metric_logger.meters['s_vit_acc1'].update(vit_student_acc1.item(), n=args.batch_size_per_gpu)

        metric_logger.meters['s_model_acc5'].update(student_acc5.item(), n=args.batch_size_per_gpu)
        metric_logger.meters['s_vit_acc5'].update(vit_student_acc5.item(), n=args.batch_size_per_gpu)

        wandb.log({"bca) test_s_model_loss": linear_student_loss_result.item(),  
                   "bcb) test_s_model_acc": linear_student_count,
                   "bda) test_s_vit_loss": linear_vit_student_loss_result.item(), 
                   "bdb) test_s_vit_acc": linear_vit_student_count, 
                })

    print(f'''Student Model Acc5 {metric_logger.s_model_acc1.global_avg} Student Model Acc5 {metric_logger.s_model_acc5.global_avg} 
            Student Vit Acc1 {metric_logger.s_vit_acc1.global_avg} Student vit Acc5 {metric_logger.s_vit_acc5.global_avg} 
            Student Model Loss {metric_logger.s_model_loss.global_avg} Student Vit Loss {metric_logger.s_vit_loss.global_avg} \n''')
   
    student_epoch_acc = float(student_acc_count / it_count) * 100
    vit_student_epoch_acc = float(vit_student_acc_count / it_count) * 100
    
    wandb.log({"baa) test_s_model_global_loss": metric_logger.s_model_loss.global_avg,
               "bab) test_s_model_global_acc": student_epoch_acc, 
               "bac) test_s_model_global_acc1": metric_logger.s_model_acc1.global_avg,
               "bad) test_s_model_global_acc5": metric_logger.s_model_acc5.global_avg,
               "bba) test_s_vit_global_loss": metric_logger.s_vit_loss.global_avg,
               "bbb) test_s_vit_global_acc": vit_student_epoch_acc,
               "bbc) test_s_vit_global_acc1": metric_logger.s_vit_acc1.global_avg,
               "bbd) test_s_vit_global_acc5": metric_logger.s_vit_acc5.global_avg
               })
        
    return None


def postprocess(student, linear_student, linear_vit_student, args):
    path_iter = os.walk(args.postprocess_path)
    root, dirs, _ = next(path_iter)
    paths = []

    for dir in dirs:
        paths.append(os.path.join(root, dir))

    transform = DataAugmentationDINO(args.global_crops_scale, args.local_crops_scale, args.local_crops_number)

    if args.postprocess_train:
        lin_s = LinearClassifier(args.out_dim, args.num_labels)
        lin_s = lin_s.cuda()
        lin_s = nn.parallel.DistributedDataParallel(lin_s, device_ids=[args.gpu])

        lin_vit_s = LinearClassifier(student.module.backbone.embed_dim, args.num_labels)
        lin_vit_s = lin_vit_s.cuda()
        lin_vit_s = nn.parallel.DistributedDataParallel(lin_vit_s, device_ids=[args.gpu])

        lin_s_opt = torch.optim.SGD(lin_s.parameters(), args.lr, momentum=0.9, weight_decay=0)
        lin_s_sch = torch.optim.lr_scheduler.CosineAnnealingLR(lin_s_opt, args.epochs, eta_min=0)

        lin_vit_s_opt = torch.optim.SGD(lin_vit_s.parameters(), args.lr, momentum=0.9, weight_decay=0)
        lin_vit_s_sch = torch.optim.lr_scheduler.CosineAnnealingLR(lin_vit_s_opt, args.epochs, eta_min=0)

        lin_s.train()
        lin_vit_s.train()

        for ep in range(1,11):

            student_acc_count = 0
            vit_student_acc_count = 0
            it_count = 0

            # dataset = datasets.Imagenet(root='/home/rushil/data', train=True, download=False, transform=transform)
            dataset = datasets.CIFAR100(root='/home/rushil/data', train=False, download=True, transform=transform)
            sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
            loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=args.batch_size_per_gpu, 
                                                 num_workers=args.num_workers, pin_memory=True, drop_last=True)

            metric_logger = utils.MetricLogger(delimiter=" ")
            header = f'Postprocess Dataset Training [{ep}/10]'

            for inp, target in metric_logger.log_every(loader, args.print_freq, header):

                inp = [im.cuda(non_blocking=True) for im in inp]
                target = target.cuda(non_blocking=True)

                target_size = target.size(0)

                with torch.no_grad():
                    student_output, student_input = student(inp)

                student_output = student_output[:target_size, :]
                student_input = student_input[:target_size, :]

                linear_student_output = lin_s(student_output)
                linear_vit_student_output = lin_vit_s(student_input)

                linear_student_loss_result = nn.CrossEntropyLoss()(linear_student_output, target)
                linear_vit_student_loss_result = nn.CrossEntropyLoss()(linear_vit_student_output, target)

                lin_s_opt.zero_grad()
                linear_student_loss_result.backward()

                lin_vit_s_opt.zero_grad()
                linear_vit_student_loss_result.backward()

                lin_s_opt.step()
                lin_vit_s_opt.step()

                linear_student_max = torch.argmax(linear_student_output, dim=-1)
                linear_student_count = torch.sum(linear_student_max == target).item()

                linear_vit_student_max = torch.argmax(linear_vit_student_output, dim=-1)
                linear_vit_student_count = torch.sum(linear_vit_student_max == target).item()

                student_acc_count = student_acc_count + linear_student_count
                vit_student_acc_count = vit_student_acc_count + linear_vit_student_count
                it_count = it_count + target_size

                linear_student_count = float(linear_student_count / target_size) * 100
                linear_vit_student_count = float(linear_vit_student_count / target_size) * 100

                student_acc1, student_acc5 = utils.accuracy(linear_student_output, target, topk=(1, 5))
                vit_student_acc1, vit_student_acc5 = utils.accuracy(linear_vit_student_output, target, topk=(1, 5))

                metric_logger.update(s_model_loss=linear_student_loss_result.item())
                metric_logger.update(s_vit_loss=linear_vit_student_loss_result.item())

                metric_logger.meters['s_model_acc1'].update(student_acc1.item(), n=args.batch_size_per_gpu)
                metric_logger.meters['s_vit_acc1'].update(vit_student_acc1.item(), n=args.batch_size_per_gpu)

                if args.num_labels >= 3:
                    metric_logger.meters['s_model_acc5'].update(student_acc5.item(), n=args.batch_size_per_gpu)
                    metric_logger.meters['s_vit_acc5'].update(vit_student_acc5.item(), n=args.batch_size_per_gpu)

                wandb.log({"dca) postprocess_s_model_loss": linear_student_loss_result.item(),  
                           "dcb) postprocess_s_model_avg_acc": linear_student_count,
                           "dda) postprocess_s_vit_loss": linear_vit_student_loss_result.item(), 
                           "ddb) postprocess_s_vit_avg_acc": linear_vit_student_count,
                           })
                
            student_epoch_acc = float(student_acc_count / it_count) * 100
            vit_student_epoch_acc = float(vit_student_acc_count / it_count) * 100

            wandb.log({"daa) postprocess_s_model_global_loss": metric_logger.s_model_loss.global_avg,
                       "dab) postprocess_s_model_global_avg_acc": student_epoch_acc, 
                       "dac) postprocess_s_model_global_acc1": metric_logger.s_model_acc1.global_avg,
                       "dad) postprocess_s_model_global_acc3": metric_logger.s_model_acc3.global_avg,
                       "dba) postprocess_s_vit_global_loss": metric_logger.s_vit_loss.global_avg,
                       "dbb) postprocess_s_vit_global_avg_acc": vit_student_epoch_acc,
                       "dbc) postprocess_s_vit_global_acc1": metric_logger.s_vit_acc1.global_avg,
                       "dbd) postprocess_s_vit_global_acc3": metric_logger.s_vit_acc3.global_avg
                       })

            lin_s_sch.step()
            lin_vit_s_sch.step()

        lin_s.eval()
        lin_vit_s.eval()

    for path in range(1,11):

        student_acc_count = 0
        vit_student_acc_count = 0
        it_count = 0

        # dataset = datasets.ImageFolder(path, transform=transform)
        # dataset = datasets.ImageNet(root='/home/rushil/data', split='val', transform=transform)
        dataset = datasets.CIFAR100(root='/home/rushil/data', train=False, download=True, transform=transform)
        subset_dataset = torch.utils.data.Subset(dataset, range((path-1)*5000, (path)*5000))
        sampler = torch.utils.data.DistributedSampler(subset_dataset, shuffle=True)
        loader = torch.utils.data.DataLoader(subset_dataset, sampler=sampler, batch_size=64, 
                                             num_workers=args.num_workers, pin_memory=True, drop_last=True)
        
        metric_logger = utils.MetricLogger(delimiter=" ")

        # header = f'Test Dataset [{path.split("/")[-1]}]'
        header = f'Test Dataset [{path}]'

        for inp, target in metric_logger.log_every(loader, args.print_freq, header):

            inp = [im.cuda(non_blocking=True) for im in inp]
            target = target.cuda(non_blocking=True)

            target_size = target.size(0)

            with torch.no_grad():
                student_output, student_input = student(inp)

            student_output = student_output[:target_size, :]
            student_input = student_input[:target_size, :]

            if args.postprocess_train:
                linear_student_output = lin_s(student_output)
                linear_vit_student_output = lin_vit_s(student_input)
            else:
                linear_student_output = linear_student(student_output)
                linear_vit_student_output = linear_vit_student(student_input)

            linear_student_loss_result = nn.CrossEntropyLoss()(linear_student_output, target)
            linear_vit_student_loss_result = nn.CrossEntropyLoss()(linear_vit_student_output, target)

            linear_student_max = torch.argmax(linear_student_output, dim=-1)
            linear_student_count = torch.sum(linear_student_max == target).item()

            linear_vit_student_max = torch.argmax(linear_vit_student_output, dim=-1)
            linear_vit_student_count = torch.sum(linear_vit_student_max == target).item()

            student_acc_count = student_acc_count + linear_student_count
            vit_student_acc_count = vit_student_acc_count + linear_vit_student_count
            it_count = it_count + target_size

            linear_student_count = float(linear_student_count / target_size) * 100
            linear_vit_student_count = float(linear_vit_student_count / target_size) * 100

            student_acc1, student_acc5 = utils.accuracy(linear_student_output, target, topk=(1, 5))
            vit_student_acc1, vit_student_acc5 = utils.accuracy(linear_vit_student_output, target, topk=(1, 5))

            metric_logger.update(s_model_loss=linear_student_loss_result.item())
            metric_logger.update(s_vit_loss=linear_vit_student_loss_result.item())

            metric_logger.meters['s_model_acc1'].update(student_acc1.item(), n=args.batch_size_per_gpu)
            metric_logger.meters['s_vit_acc1'].update(vit_student_acc1.item(), n=args.batch_size_per_gpu)

            metric_logger.meters['s_model_acc5'].update(student_acc5.item(), n=args.batch_size_per_gpu)
            metric_logger.meters['s_vit_acc5'].update(vit_student_acc5.item(), n=args.batch_size_per_gpu)

            wandb.log({"cca) postprocess_s_model_loss": linear_student_loss_result.item(),  
                       "ccb) postprocess_s_model_acc": linear_student_count,
                       "cda) postprocess_s_vit_loss": linear_vit_student_loss_result.item(), 
                       "cdb) postprocess_s_vit_acc": linear_vit_student_count,
                       })

        print(f'''Student Model Acc1 {metric_logger.s_model_acc1.global_avg} Student Model Acc3 {metric_logger.s_model_acc3.global_avg} 
            Student Vit Acc1 {metric_logger.s_vit_acc1.global_avg} Student vit Acc3 {metric_logger.s_vit_acc3.global_avg} 
            Student Model Loss {metric_logger.s_model_loss.global_avg} Student Vit Loss {metric_logger.s_vit_loss.global_avg} \n''')
    
        student_epoch_acc = float(student_acc_count / it_count) * 100
        vit_student_epoch_acc = float(vit_student_acc_count / it_count) * 100
        
        wandb.log({"caa) postprocess_s_model_global_loss": metric_logger.s_model_loss.global_avg,
                  "cab) postprocess_s_model_global_acc": student_epoch_acc, 
                  "cac) postprocess_s_model_global_acc1": metric_logger.s_model_acc1.global_avg,
                  "cad) postprocess_s_model_global_acc3": metric_logger.s_model_acc3.global_avg,
                  "cba) postprocess_s_vit_global_loss": metric_logger.s_vit_loss.global_avg,
                  "cbb) postprocess_s_vit_global_acc": vit_student_epoch_acc,
                  "cbc) postprocess_s_vit_global_acc1": metric_logger.s_vit_acc1.global_avg,
                  "cbd) postprocess_s_vit_global_acc3": metric_logger.s_vit_acc3.global_avg
                  })
        
    return None
            

class LinearClassifier(nn.Module):
    def __init__(self, dim, num_labels):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()

        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))

        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])

        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])

        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


class NormalizedDataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_transfo1 = transforms.Compose([
            transforms.CenterCrop(224),
            normalize,
        ])

        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            normalize,
        ])

        self.global_transfo3 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])

        self.global_transfo4 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])

        self.global_transfo5 = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.1),
            normalize,
        ])

        self.global_transfo6 = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            normalize,
        ])

        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        crops.append(self.global_transfo3(image))
        crops.append(self.global_transfo4(image))
        crops.append(self.global_transfo5(image))
        crops.append(self.global_transfo6(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)

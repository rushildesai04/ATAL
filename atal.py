# ATAL Model - Always Testing, Always Learning
# Developed By: Rushil Desai
# 
# Base model: DINO - Self Supervised System By Facebook AI
# Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0

# // Run to make sure DINO works without editing code
# // Make linear classifier on top to output labels for images
# // Edit data loader to feed in 1 image at a time
# // Keep track of loss, accuracy
# // Test each image at the begining with multiple epochs
# // Write the main training loop

import argparse
import os
import sys
import time
import math
import json
import datetime
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
import kckl
from vision_transformer import ATALHead

import wandb
import ipdb

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():

    parser = argparse.ArgumentParser('ATAL', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the ATAL head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the ATAL head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=5, type=int,
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
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs of training.')
    parser.add_argument('--image_iterations', default=10, type=int, help="""Number of times image 
        should be trained on in an epoch.""")
    parser.add_argument("--warmup_epochs", default=5, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--min_atal_loss', default='1.0', type=str,
        help="""Minimum loss required to finish training epoch""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.5, 1.0),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=4, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.1, 0.5),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default="/home/rushil/data/clear-10-train/labeled_images/1", type=str, 
        help='Path to save logs and checkpoints.')
    parser.add_argument('--output_dir', default="/home/rushil/ATAL/outputs/output_atal", type=str, 
        help='Path to the dataset.')
    parser.add_argument('--epoch_print_freq', default=10, type=int, help="""Print epoch stats 
        at every interval.""")
    parser.add_argument('--saveckp_freq', default=1, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    return parser


def TrainATAL(args):

    # ============ init system ... ============

    if not dist.is_initialized():
        utils.init_distributed_mode(args)

    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============

    transform = DataAugmentationATAL(args.global_crops_scale, args.local_crops_scale, args.local_crops_number)

    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=1, num_workers=args.num_workers,
                                              pin_memory=True, drop_last=True)

    print(f"Data loaded with {len(dataset)} images.")

    # ============ building student and teacher networks ... ============

    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](patch_size=args.patch_size, drop_path_rate=args.drop_path_rate) #stochastic depth
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, ATALHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(teacher, ATALHead(
        embed_dim, 
        args.out_dim, 
        use_bn=args.use_bn_in_head,
    ))

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()

    # device_ids=[os.environ.get('CUDA_VISIBLE_DEVICES')]
    # device_ids = ['cuda:' + id for id in device_ids]
    device_ids = [args.gpu]

    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=device_ids, find_unused_parameters=True)
        teacher_without_ddp = teacher.module

    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher

    student = nn.parallel.DistributedDataParallel(student, device_ids=device_ids, find_unused_parameters=True)
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())

    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"Student and Teacher are built with the {args.arch} network.")

    # ============ building KCKL classifier ... ============

    classifier = kckl.KCKL(data_path=args.data_path, embed_dim=args.out_dim, gpu=args.gpu)
    classifier.build_network()

    # ============ preparing loss ... ============
    atal_loss = LossATAL(
        args.out_dim,
        args.local_crops_number + 4,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches

    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============

    lr_schedule = utils.cosine_scheduler(
        args.lr * (len(data_loader) * utils.get_world_size()) / 256.,  # linear scaling rule, 1 signifies 1 image per GPU
        args.min_lr,
        args.epochs, 
        len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )

    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, 
        len(data_loader),
    )

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(data_loader))
    iteration_schedule = lambda x: int(args.image_iterations - (args.image_iterations - 1) * (1 - np.exp(-x / (args.image_iterations**2))))
    # iteration_schedule = lambda x: 1

    print(f"Loss, Optimizer and Schedulers are ready.")

    wandb.init(
        project="ATAL",
        config={
        "learning_rate": lr_schedule,
        "weight_decay": wd_schedule,
        "momentum": momentum_schedule,
        "architecture": "ATAL",
        "dataset": "CLEAR-10",
        "epochs": args.epochs,
        }
    )

    # ============ optionally resume training ... ============

    to_restore = {"epoch": 0, "correct_count": 0, "iterations": 0}

    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "atal_ckpt.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        atal_loss=atal_loss,
    )

    start_epoch = to_restore["epoch"]
    correct_count = to_restore["correct_count"]
    iterations = to_restore["iterations"]

    start_time = time.time()
    print("Starting ATAL training !")
    
    for epoch in range(start_epoch, args.epochs):

        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of ATAL ... ============

        train_stats, train_corr, train_it = OneEpochATAL(student, teacher, teacher_without_ddp, classifier, atal_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule, iteration_schedule,
            epoch, fp16_scaler, args, correct_count, iterations)

        # ============ writing logs ... ============

        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            "correct_count": train_corr,
            'iterations': train_it,
            'atal_loss': atal_loss.state_dict(),
            'args': args,
        }

        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()

        correct_count = train_corr
        iterations = train_it

        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'atal_ckpt.pth'))

        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'atal_ckpt_{epoch + 1}.pth'))
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        if utils.is_main_process():
            current_datetime = datetime.datetime.now()
            current_datetime = current_datetime.strftime("%m/%d/%Y %H:%M:%S")
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(f'(ATAL) Epoch: [{epoch + 1}/{args.epochs}] Time: [{current_datetime}] --> ' 
                        + json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print('Training time {}'.format(total_time_str))
    wandb.finish()


def OneEpochATAL(student, teacher, teacher_without_ddp, classifier, atal_loss, data_loader, optimizer, lr_schedule, 
                    wd_schedule, momentum_schedule, iteration_schedule, epoch, fp16_scaler, args, corr, it):

    iteration_range = iteration_schedule(epoch)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch + 1}/{args.epochs}]'
    args.epoch_print_freq = len(data_loader)-1

    acc_count = 0
    corr_count = corr
    it_count = it
    
    for batch, (image, target) in enumerate(metric_logger.log_every(data_loader, args.epoch_print_freq, header)):

        for _ in range(iteration_range):

            # global training iteration
            it = (len(data_loader) * epoch) + batch

            # update weight decay and learning rate according to their schedule
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]

                # only the first group is regularized 
                if i == 0:
                    param_group["weight_decay"] = wd_schedule[it]
            
            # move images to gpu
            image = [i.cuda(non_blocking=True) for i in image]

            # teacher and student forward passes + compute atal loss
            with torch.cuda.amp.autocast(fp16_scaler is not None):
                teacher_output = teacher(image)
                student_output = student(image)

                loss = atal_loss(student_output, teacher_output, epoch)

            # stop training if loss is not finite
            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), force=True)
                sys.exit(1)

            # student update
            optimizer.zero_grad()

            if fp16_scaler is None:
                loss.backward()
                if args.clip_grad:
                    param_norms = utils.clip_gradients(student, args.clip_grad)
                else:
                    param_norms = None
                utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
                optimizer.step()
            
            else:
                fp16_scaler.scale(loss).backward()
                if args.clip_grad:
                    fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    param_norms = utils.clip_gradients(student, args.clip_grad)
                else:
                    param_norms = None
                utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            with torch.no_grad():
                m = momentum_schedule[it]  # momentum parameter
                for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            classification_image = student_output.detach()
            classification_target = target.detach()
            
            _, class_pred, class_tar, class_pred_cache, class_tar_cache = classifier.eval_network(
                classification_image, classification_target)
            
            acc1, acc3 = utils.accuracy(class_pred, class_tar, topk=(1, 3))

            if class_pred_cache[0] == class_tar_cache[0]:
                acc_count = acc_count + 1
                corr_count = corr_count + 1
            it_count = it_count + 1
            main_acc = float(corr_count / it_count) * 100

            # logging
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(acc=main_acc)
            metric_logger.meters['acc1'].update(acc1.item(), n=iteration_range)
            metric_logger.meters['acc3'].update(acc3.item(), n=iteration_range)
            metric_logger.update(corr=acc_count)
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
            wandb.log({"acc_local": main_acc, "acc1_local": acc1.item(), "acc3_local": acc3.item(), 
                       "corr": corr_count, "it": it_count})

        # gather the stats from the batch
        metric_logger.synchronize_between_processes()
        print(f'Batch [{batch + 1}/{len(data_loader)}] Average Stats:', metric_logger)

        if atal_loss.item().item() <= float(args.min_atal_loss):
            print(f'Stopping Epoch [{epoch + 1}/{args.epochs}] ATAL training because loss has reached a sufficient value')
            break

        wandb.log({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"], "wd": optimizer.param_groups[0]["weight_decay"], 
                   "acc_general": main_acc, "acc1_general": acc1.item(), "acc3_general": acc3.item()})


    # gather the stats from all batches
    metric_logger.synchronize_between_processes()
    print(f'Epoch [{epoch + 1}/{args.epochs}] Stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, corr_count, it_count


class LossATAL(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.curr_loss = 0
        self.register_buffer("center", torch.zeros(1, out_dim))

        # apply a warm up for the teacher temperature because a high temperature makes the training instable at beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        student_out = student_output / self.student_temp
        # student_out = F.log_softmax(student_out, dim=-1)
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(4)
        
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
        self.curr_loss = total_loss
        # total_loss = nn.CrossEntropyLoss()(student_out, teacher_out)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def item(self):
        return self.curr_loss


class DataAugmentationATAL(object):
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

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.CenterCrop(224),
            normalize,
        ])

        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            normalize,
        ])

        # third global crop
        self.global_transfo3 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])

        # fourth global crop
        self.global_transfo4 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])

        # transformation for the local small crops
        self.local_crops_number = local_crops_number

        # first local crop
        self.local_transfo1 = transforms.Compose([
            transforms.RandomCrop(96),
            normalize,
        ])

        # second local crop
        self.local_transfo2 = transforms.Compose([
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
        for _ in range(self.local_crops_number // 2):
            crops.append(self.local_transfo1(image))
            crops.append(self.local_transfo2(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ATAL', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    TrainATAL(args)


# *** PSEUDOCODE ***


# def train_one_epoch(student, teacher, teacher_without_ddp, atal_loss, data_loader,
#                     optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
#                     fp16_scaler, args):
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
#     for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
#         # update weight decay and learning rate according to their schedule
#         it = len(data_loader) * epoch + it  # global training iteration
#         for i, param_group in enumerate(optimizer.param_groups):
#             param_group["lr"] = lr_schedule[it]
#             if i == 0:  # only the first group is regularized
#                 param_group["weight_decay"] = wd_schedule[it]
#
#         # move images to gpu
#         images = [im.cuda(non_blocking=True) for im in images]
#         # teacher and student forward passes + compute atal loss
#         with torch.cuda.amp.autocast(fp16_scaler is not None):
#
#             for internal_iter_ssl in args.num_internal_iters_ssl  # another option is until loss is smaller than something
#                 teacher_input, student_input = get_augments(images)  # apply new augs here instead of in data loader
#
#                 teacher_output = teacher(teacher_input)
#                 student_output = student(student_input)
#                 loss_ssl = atal_loss(student_output, teacher_output, epoch)
#
#                 optimizer.zero_grad()  # updating the student
#                 loss_ssl.backward()
#                 optimizer.step()   #  check about technical updating down the atal code becaue AMP
#
#                 if loss_ssl < args.min_loss_ssl:
#                     break
#
#             prediction = classifier(student_output)
#             # here you log/print/keep track of predictions and their certainties
#
#             for internal_iter_supervised in args.num_internal_iters_supervised  # another option is until loss is smaller than something
#                 prediction = classifier(student_output)
#                 loss_supervised = criterion(prediction)  # use cross entropy I guess
#
#                 optimizer.zero_grad()  # updating the student
#                 loss_ssl.backward()
#                 optimizer.step()   #  check about technical updating down the atal code becaue AMP
#
#                 if loss_ssl < args.min_loss_supervised:
#                     break
#
#             # update the teacher with EMA


# *** GENERATED DESCRIPTION ***


# This code is a Python script for training a self-supervised learning model called ATAL (Always Testing, Always Learning)
# for computer vision tasks. ATAL is a method used to pretrain deep neural networks on large datasets without the need
# for manual annotations. Let's break down the code step by step and explain each function's purpose:

# ### Importing Libraries

# - The code begins by importing various Python libraries and modules. These include essential libraries like `argparse` for 
# command-line argument parsing, `torch` for PyTorch-based deep learning, and several other utility and computer vision-related 
# libraries.

# ### Command-Line Arguments

# - The `get_args_parser` function defines command-line arguments for configuring the ATAL training process. These arguments 
# control various aspects of the training, such as model architecture, data augmentation, optimization parameters, and more.

# ### Training Function

# - The `train_atal` function is the main function responsible for training the ATAL model. It performs the following steps:
#     1. Initializes distributed training if multiple GPUs are used.
#     2. Sets random seeds to ensure reproducibility.
#     3. Prints information about the Git commit hash and training parameters.
#     4. Enables benchmarking mode for the CuDNN library (for GPU optimization).
#     5. Prepares the dataset and data loader for training using PyTorch's `ImageFolder` and `DataLoader` classes.
#     6. Builds the student and teacher neural networks. The architecture of these networks can be selected via command-line 
#        arguments, and these networks are used to perform self-supervised learning.
#     7. Defines the ATAL loss function, which measures the similarity between the outputs of the student and teacher networks.
#     8. Sets up the optimizer for model training, such as AdamW, SGD, or LARS.
#     9. Initializes a learning rate schedule, weight decay schedule, and momentum schedule.
#     10. Optionally, resumes training from a checkpoint if specified.
#     11. Starts training for a specified number of epochs.
#     12. For each epoch, it iterates through the data loader, calculates and updates gradients, and performs teacher-student 
#         interaction to minimize the loss.
#     13. Logs training statistics, including loss, learning rate, and weight decay.
#     14. Saves checkpoints and logs to the specified output directory.

# ### Loss Function

# - The `LossATAL` class defines the ATAL loss function. It measures the cross-entropy between the softmax outputs of the teacher 
# and student networks. The loss function also incorporates temperature scaling and centering of the teacher's output for 
# improved training stability.

# ### Data Augmentation

# - The `DataAugmentationATAL` class defines data augmentation transformations for image data. It includes random resizing, 
# flipping, color jittering, Gaussian blur, and other augmentations. These augmentations are applied to both global and local 
# crops of the input image.

# ### Main Block

# - Finally, the code checks if it's being run as the main script. If so, it parses command-line arguments, creates the output 
# directory if it doesn't exist, and calls the `train_atal` function to start the training process.

# In summary, this code is designed for self-supervised training of deep neural networks using the ATAL method. It provides a 
# flexible set of command-line arguments to configure various aspects of the training process, including model architecture, 
# data augmentation, and optimization parameters. The code is suitable for training on large image datasets without the need 
# for manually labeled data, making it valuable for computer vision applications.

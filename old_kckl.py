# KCKL Classifier - Keep Classifying, Keep Learning
# Developed By: Rushil Desai
# 
# Base model: PyTorch - Self Supervised System By Facebook AI

# // Create Classifier to take input image and ouptut predicted label

import argparse
import os
import sys
import time
import subprocess
import math
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import ATAL.old_atal as old_atal
import utils
import vision_transformer as vits

import wandb
import ipdb

class KCKL():

    def __init__(self, arch='vit_small', data_path='/home/rushil/data/clear-10-train/labeled_images/1',
                 output_dir='/home/rushil/ATAL/outputs/output_kckl', checkpoint_key='kckl', pretrained_weights='', 
                 dist_url='env://', num_workers=16, batch_size_per_gpu=1, lr=0.00001, min_kckl_loss=0.0001, patch_size=16, 
                 n_last_blocks=1, warmup_epochs=10, embed_dim=65536, concat_dim=True, saveckpt_freq=20, 
                 print_freq=10, seed=0, gpu=None):
        
        # class parameters
        self.arch = arch
        self.data_path = data_path
        self.output_dir = output_dir
        self.checkpoint_key = checkpoint_key
        self.pretrained_weights = pretrained_weights
        self.dist_url = dist_url
        self.num_workers = num_workers
        self.batch_size_per_gpu = batch_size_per_gpu
        self.lr = lr
        self.min_kckl_loss = min_kckl_loss
        self.patch_size = patch_size
        self.n_last_blocks = n_last_blocks
        self.warmup_epochs = warmup_epochs
        self.embed_dim = embed_dim
        self.concat_dim = concat_dim
        self.saveckpt_freq = saveckpt_freq
        self.print_freq = print_freq
        self.seed = seed
        self.gpu = gpu
        self.input_dim = None
        self.linear_input_dim = None
        self.num_labels = None
        self.data_classes = None
        self.model = None
        self.classifier = None
        self.head = None
        self.loss = None
        self.optimizer = None

        parser = argparse.ArgumentParser('KCKL Classifier')
        parser.add_argument('--arch', default=self.arch, type=str)
        parser.add_argument('--data_path', default=self.data_path, type=str)
        parser.add_argument('--output_dir', default=self.output_dir, type=str)
        parser.add_argument("--checkpoint_key", default=self.checkpoint_key, type=str)
        parser.add_argument('--pretrained_weights', default=self.pretrained_weights, type=str)
        parser.add_argument("--dist_url", default=self.dist_url, type=str)
        parser.add_argument('--num_workers', default=self.num_workers, type=int)
        parser.add_argument('--batch_size_per_gpu', default=self.batch_size_per_gpu, type=int)
        parser.add_argument("--lr", default=self.lr, type=float)
        parser.add_argument("--min_kckl_loss", default=self.min_kckl_loss, type=float)
        parser.add_argument('--patch_size', default=self.patch_size, type=int)
        parser.add_argument('--n_last_blocks', default=self.n_last_blocks, type=int)
        parser.add_argument('--warmup_epochs', default=self.warmup_epochs, type=int)
        parser.add_argument('--concat_dim', default=self.concat_dim, type=bool)
        parser.add_argument('--saveckpt_freq', default=self.saveckpt_freq, type=int)
        parser.add_argument('--print_freq', default=self.print_freq, type=int)
        parser.add_argument("--local_rank", default=0, type=int)        
        self.args = parser.parse_args()


    def build_network(self):

        # Init model and system info
        if not dist.is_initialized():
            utils.init_distributed_mode(self.args)
            
        print("git:\n  {}\n".format(utils.get_sha()))
        print("\n".join("%s: %s" % (k, str(v)) for k, v in self.__dict__.items()))
        cudnn.benchmark = True

        # If the network is a Vision Transformer (vit_tiny, vit_small, vit_base, vit_large)
        if self.arch in vits.__dict__.keys():
            model = vits.__dict__[self.arch](patch_size=self.patch_size, num_classes=0)
        else:
            print(f"Unknow architecture: {self.arch}")
            sys.exit(1)

        # Move model to GPU hardware
        model.cuda()
        model.eval()
        self.model = model

        self.head = KCKLHead(self.model.embed_dim, self.embed_dim, use_bn=False, norm_last_layer = False)

        # Load weights to evaluate
        utils.load_pretrained_weights(model, self.pretrained_weights, self.checkpoint_key, self.arch, self.patch_size)
        print(f"Model {self.arch} built.")

        # Fetch number of dataset labels
        try:
            num_labels = subprocess.run("ls {} | wc -l".format(self.data_path), shell=True, check=True, stdout=subprocess.PIPE)
            # {self.data_path}
            self.num_labels = int(num_labels.stdout.decode('utf-8').strip())
        except:
            self.num_labels = 0

        # Establish input dimension
        if self.arch == 'vit_tiny':
            self.input_dim = 16250944
            self.linear_input_dim = self.embed_dim
        elif self.arch == 'vit_small':
            self.input_dim = 16250944
            self.linear_input_dim = self.embed_dim
        elif self.arch == 'vit_base':
            self.input_dim = 16250944
            self.linear_input_dim = self.embed_dim
        elif self.arch == 'vit_large':
            self.input_dim = 16250944
            self.linear_input_dim = self.embed_dim
        else:
            self.input_dim = 0
            self.linear_input_dim = 0

        # Make Network
        if self.concat_dim:
            self.classifier = Network2D(self.linear_input_dim, self.num_labels)
        else:
            self.classifier = Network3D(self.input_dim, self.num_labels)

        self.classifier = self.classifier.cuda()

        if self.gpu == None:
            device_ids=[os.environ.get('CUDA_VISIBLE_DEVICES')]
            device_ids = ['cuda:' + id for id in device_ids]
        else:
            device_ids = [self.gpu]

        self.classifier = nn.parallel.DistributedDataParallel(self.classifier, device_ids=device_ids)

        # Define Loss and Optimizer
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.classifier.parameters(), self.lr)

        # Fetch dataset classes and put them into a set
        self.data_classes = [f for f in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, f))]
        self.data_classes = sorted(self.data_classes)
        self.data_classes = tuple(self.data_classes)

        # Transform images to tensors
        train_transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # create dataset of images
        dataset_train = datasets.ImageFolder(self.data_path, transform=train_transform)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)

        # create loader for dataset
        train_loader = torch.utils.data.DataLoader(dataset_train, sampler=sampler, batch_size=self.batch_size_per_gpu,
            num_workers=self.num_workers, pin_memory=True)
        
        # log loaded data
        print(f"Data loaded with {len(dataset_train)} train images.")

        wandb.init(
            project="KCKL",
            config={
            "classifier": self.classifier,
            "loss": self.loss,
            "optimizer": self.optimizer,
            "transform": "KCKL",
            "dataset": "CLEAR-10",
            "epochs": self.warmup_epochs,
            }
        )

        # resume from a checkpoint
        to_restore = {"epoch": 0, "best_acc": 0.0}
        utils.restart_from_checkpoint(
            os.path.join(self.output_dir, "kckl_ckpt.pth"), run_variables=to_restore,
            state_dict=self.classifier, optimizer=self.optimizer)

        # restore variables
        start_epoch = to_restore["epoch"]
        best_acc = to_restore["best_acc"]

        # training loop for kckl model
        for epoch in range(start_epoch, self.warmup_epochs):

            # set training epoch
            train_loader.sampler.set_epoch(epoch)

            # run one training epoch on model
            train_stats = train(self.model, self.head, self.classifier, self.optimizer, train_loader, epoch, self.min_kckl_loss, 
                self.n_last_blocks, self.warmup_epochs, self.concat_dim, self.print_freq)

            # create save info for model
            save_dict = {"epoch": epoch + 1, "state_dict": self.classifier.state_dict(), 
                         "optimizer": self.optimizer.state_dict(), "best_acc": best_acc,}

            # save model info
            torch.save(save_dict, os.path.join(self.output_dir, "kckl_ckpt.pth"))

            # save specific model info at checkpoint
            torch.save(save_dict, os.path.join(self.output_dir, f'kckl_ckpt_{epoch+1}.pth'))

            # create log info for training epoch
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

            # write info to logs
            if utils.is_main_process():
                current_datetime = datetime.now()
                current_datetime = current_datetime.strftime("%m/%d/%Y %H:%M:%S")
                with (Path(self.output_dir) / "log.txt").open("a") as f:
                    f.write(f'(KCKL) Epoch: [{epoch + 1}/{self.warmup_epochs}] Time: [{current_datetime}] --> ' 
                            + json.dumps(log_stats) + "\n")
        
        # print success message
        print(f'Finished Loading KCKL Classifier For {self.warmup_epochs} Epochs on {str(len(train_loader))} Images!')
        wandb.finish()
    
    def eval_network(self, image, target):
        return validate(self.model, self.classifier, self.optimizer, image, target, self.num_workers, self.batch_size_per_gpu, 
                        self.n_last_blocks, self.concat_dim, self.data_classes, self.print_freq)


def train(model, head, classifier, optimizer, loader, epoch, min_kckl_loss, n, warmup_epochs, concat_dim, print_freq):

    # set classifier in train mode
    classifier.train()

    # set up logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    # define variables needed for outputting training statistics
    header = f'(KCKL) Epoch: [{epoch + 1}/{warmup_epochs}] Iteration: '

    # training loop for all data in dataset
    for (inp, target) in metric_logger.log_every(loader, print_freq, header):

        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():

            intermediate_output = model.get_intermediate_layers(inp, n)
            intermediate_output = [head(x) for x in intermediate_output]

            if concat_dim:
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            else:
                output = torch.cat([x for x in intermediate_output])
                

        # run classifier on output
        output = classifier(output)

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        wandb.log({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})

        if loss.item() <= min_kckl_loss:
            print(f'Stopping Epoch [{epoch + 1}/{warmup_epochs}] KCKL training because loss has reached a sufficient value')
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged Epoch Stats:", metric_logger)

    # return training stats
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate(model, classifier, optimizer, image, target, num_workers, batch_size_per_gpu, n, concat_dim, data_classes, 
             print_freq):

    # create validation loader
    val_loader = torch.utils.data.DataLoader(image, batch_size=batch_size_per_gpu, num_workers=num_workers, pin_memory=True)

    # set classifier in eval mode
    classifier.eval()

    # set up logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    prediction_cache = []
    target_cache = []
    header = '(KCKL) Testing...'

    # move to gpu
    inp = image.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)

    # run classifier on output
    output = classifier(inp)
    output = torch.mean(output, dim=0, keepdim=True)

    # get prediction from classifier output
    prediction = nn.Softmax(output)
    prediction = torch.argmax(output).item()
    prediction = data_classes[prediction]
    prediction_cache.append(prediction)

    # convert target data label to string
    target_label = data_classes[target.item()]
    target_cache.append(target_label)

    # compute cross entropy loss
    loss = nn.CrossEntropyLoss()(output, target)

    # compute the gradients
    optimizer.zero_grad()
    loss.backward()

    # step
    optimizer.step()

    # calculates top label and top 5 labels if dataset has >= 5 labels
    if len(data_classes) >= 5:

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
    else:
        
        acc1 = utils.accuracy(output, target, topk=(1,))

    # update top label log
    torch.cuda.synchronize()
    batch_size = inp.shape[0]
    metric_logger.update(loss=loss.item())
    metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    # update top 5 labels log if applicable
    if classifier.module.num_labels >= 5:
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # # eval loop for input data image
    # for inp in metric_logger.log_every(val_loader, print_freq, header):
         
    # output eval stats produced by the classifier
    if len(data_classes) >= 5:
        print(f'Top 1: [{metric_logger.acc1.global_avg:.3f}] Top 5: [{metric_logger.acc5.global_avg:.3f}] Loss: [{metric_logger.loss.global_avg:.3f}] Prediction: {prediction_cache} Target: {target_cache}')
    else:
        print(f'Top 1: [{metric_logger.acc1.global_avg:.3f}] Loss: [{metric_logger.loss.global_avg:.3f}] Prediction: {prediction_cache} Target: {target_cache}')

    # return eval stats
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, output, target, prediction_cache, target_cache


class KCKLHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256, num_labels=11):

        super().__init__()

        self.linear_classifier = nn.Linear(out_dim, num_labels)
        self.linear_loss = nn.CrossEntropyLoss()
        
        nlayers = max(nlayers, 1)

        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp = self.mlp.to(device)

        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_layer = self.last_layer.to(device)

        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            utils.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def linear_forward(self, h, y):
        y_h_pred = self.linear_classifier(h)
        loss_y_h = self.linear_loss(y_h_pred, y)
        return loss_y_h

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class Network2D(nn.Module):

    # Initialize the network
    def __init__(self, linear_input_dim, num_labels):

        self.linear_input_dim = linear_input_dim
        self.num_labels = num_labels
        
        super(Network2D, self).__init__()

        self.linear = nn.Linear(linear_input_dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    # The forward propagation algorithm
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        
        return x
    

class Network3D(nn.Module):

    # Initialize the network
    def __init__(self, input_dim, num_labels):

        self.input_dim = input_dim
        self.num_labels = num_labels
        
        super(Network3D, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3)

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_labels)

    # The forward propagation algorithm
    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        dim1 = x.shape[1]
        dim2 = x.shape[2]

        x = x.view(-1, 16 * dim1 * dim2)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

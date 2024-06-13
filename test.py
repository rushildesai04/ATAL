# import torch
# from PIL import Image
# from kckl import KCKL
# import ipdb
# from torchvision import datasets, transforms

# classifier = KCKL()
# classifier.build_network()

# train_transform = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         ])
# image = Image.open('/home/rushil/data/clear-10-train/labeled_images/2/bus/2351623905.jpg')
# image_tensor = train_transform(image).unsqueeze(0)
# ipdb.set_trace()
# k = classifier.eval_network(image_tensor, torch.Tensor([2]))
# ipdb.set_trace()


# import torch

# # define true labels and predicted logits as PyTorch Tensors
# y_true = torch.LongTensor([8])
# y_logits = torch.Tensor([[-0.2983, -0.1385, -0.5163, -0.0813, -0.1824,  0.3216,  0.2724, -0.3193, -0.0730, -0.0528, -0.0836]])

# # create the loss object
# ce_loss = torch.nn.CrossEntropyLoss()

# # calculate the loss
# loss = ce_loss(y_logits, y_true)

# # print the loss
# print(loss.item())

# import ipdb
# import numpy as np

# paths = ['/home/rushil/data/TVSHOWS/HARRYPOTTER/phalp//_DEMO/S003E000_SCENE_020/img/000094.jpg', 
#          '/home/rushil/data/TVSHOWS/HARRYPOTTER/phalp//_DEMO/S003E000_SCENE_020/img/000095.jpg']

# new_paths = ["/" + x.split("/")[1] + "/" + x.split("/")[2] + "/" + x.split("/")[3] + "/MOVIES/" + 
#              x.split("/")[5] + "/" + x.split("/")[6] + "/" + x.split("/")[8] + "/" + x.split("/")[9] 
#              + "/" + x.split("/")[10] + "/" + x.split("/")[11] for x in paths]

# print(new_paths)

# ipdb.set_trace()
# iteration_schedule = lambda x: max(int(10 * np.exp(-x / (400 / 2))), 1)


# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import datasets, transforms
# import random
# # from PIL import Image
# import utils
# import ipdb

# class CIFAR100Subset(Dataset):
#     def __init__(self, root, train, download, transform, classes):
#         self.cifar100 = datasets.CIFAR100(root=root, train=train, download=download, transform=transform)
#         self.classes = classes
#         self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
#         self.subset_indices = self._get_subset_indices()

#     def _get_subset_indices(self):
#         subset_indices = []
#         for idx, (_, label) in enumerate(self.cifar100):
#             if self.cifar100.classes[label] in self.classes:
#                 subset_indices.append(idx)
#             # print(f"{idx}: {label}", end=" | ")
#         return subset_indices

#     def __len__(self):
#         return len(self.subset_indices)

#     def __getitem__(self, idx):
#         subset_idx = self.subset_indices[idx]
#         img, label = self.cifar100[subset_idx]
#         return img, self.class_to_idx[self.cifar100.classes[label]]

# class SequentialClassLoader:
#     def __init__(self, root, train, download, transform, batch_size, num_load):
#         self.root = root
#         self.train = train
#         self.download = download
#         self.transform = transform
#         self.batch_size = batch_size
#         self.num_load = num_load

#         self.cifar100 = datasets.CIFAR100(root=root, train=train, download=download)
        
#         self.classes = self.cifar100.classes
#         random.shuffle(self.classes)

#         self.num_classes = len(self.classes)
#         self.current_class_subset = 0

#         self.classes_trained = []
#         self.classes_left = self.classes.copy()
#         self.classes_current = []

#     def get_dataloader(self):
#         start_idx = self.current_class_subset * self.num_load
#         end_idx = start_idx + self.num_load

#         if start_idx >= self.num_classes:
#             return None

#         self.current_class_subset = self.current_class_subset + 1

#         if end_idx > self.num_classes:
#             self.classes_current = self.classes[start_idx:self.num_classes]
#         else:
#             self.classes_current = self.classes[start_idx:end_idx]

#         self.classes_trained.extend(self.classes_current)
#         self.classes_left = self.classes_left[self.num_load:]
        
        
#         dataset = CIFAR100Subset(root=self.root, train=self.train, download=self.download, transform=self.transform, 
#                                  classes=self.classes_current)
#         dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

#         return dataloader
    
# class IncrementalClassLoader:
#     def __init__(self, root, train, download, transform, batch_size, num_load):
#         self.root = root
#         self.train = train
#         self.download = download
#         self.transform = transform
#         self.batch_size = batch_size
#         self.num_load = num_load

#         self.cifar100 = datasets.CIFAR100(root=root, train=train, download=download)
        
#         self.classes = self.cifar100.classes
#         random.shuffle(self.classes)

#         self.num_classes = len(self.classes)
#         self.current_class_subset = 0

#         self.classes_trained = []
#         self.classes_left = self.classes.copy()
#         self.classes_current = []

#     def get_dataloader(self):
#         start_idx = self.current_class_subset * self.num_load
#         end_idx = start_idx + self.num_load
        
#         if start_idx >= self.num_classes:
#             return None
        
#         self.current_class_subset = self.current_class_subset + 1
#         self.classes_current = self.classes[start_idx:end_idx]
#         self.classes_trained.extend(self.classes_current)
#         self.classes_left = self.classes_left[self.num_load:]
        
#         dataset = CIFAR100Subset(root=self.root, train=self.train, download=self.download, transform=self.transform, 
#                                  classes=self.classes_trained)
#         dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

#         return dataloader
    
# class DataAugmentationDINO(object):
#     def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
#         flip_and_color_jitter = transforms.Compose([
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomApply(
#                 [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
#                 p=0.8
#             ),
#             transforms.RandomGrayscale(p=0.2),
#         ])
#         normalize = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         ])

#         self.global_transfo1 = transforms.Compose([
#             transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
#             flip_and_color_jitter,
#             utils.GaussianBlur(1.0),
#             normalize,
#         ])

#         self.global_transfo2 = transforms.Compose([
#             transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
#             flip_and_color_jitter,
#             utils.GaussianBlur(0.1),
#             utils.Solarization(0.2),
#             normalize,
#         ])

#         self.local_crops_number = local_crops_number
#         self.local_transfo = transforms.Compose([
#             transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
#             flip_and_color_jitter,
#             utils.GaussianBlur(p=0.5),
#             normalize,
#         ])

#     def __call__(self, image):
#         crops = []
#         crops.append(self.global_transfo1(image))
#         crops.append(self.global_transfo2(image))
#         for _ in range(self.local_crops_number):
#             crops.append(self.local_transfo(image))
#         return crops

# root = '/home/rushil/data'
# # transform = DataAugmentationDINO((0.4, 1.), (0.05, 0.4), 8)
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# random.seed(random.randint(-1000000000, 1000000000))
# classloader = SequentialClassLoader(root=root, train=True, download=False, transform=transform, batch_size=64, num_load=10)

# dataloader = classloader.get_dataloader()
# while dataloader is not None:
#     for batch_idx, (data, target) in enumerate(dataloader):
#         print(f"[{batch_idx}] --> {target}")
#     print(f"New Classes --> {classloader.classes_current}")
#     print(f"Classes Trained --> {classloader.classes_trained}")
#     dataloader = classloader.get_dataloader()



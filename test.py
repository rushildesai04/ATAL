import torch
from PIL import Image
from kckl import KCKL
import ipdb
from torchvision import datasets, transforms

classifier = KCKL()
classifier.build_network()

train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
image = Image.open('/home/rushil/data/clear-10-train/labeled_images/2/bus/2351623905.jpg')
image_tensor = train_transform(image).unsqueeze(0)
ipdb.set_trace()
k = classifier.eval_network(image_tensor)
ipdb.set_trace()




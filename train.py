import torch
import torch.nn as nn
import argparse
import torchvision
import os
import random
from PIL import Image
from tqdm import tqdm

BATCH_SIZE = 8
num_worker = 4
MAX_EPOCH = 256
LR = 1e-3  # 0.0001

from Network import HS_Model
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def default_loader(path):
    from torchvision import get_image_backend
    return pil_loader(path)

class MvtecDataLoader(torch.utils.data.Dataset):

    # constructor of the class
    def __init__(self, path, transform, normal_number=0, shuffle=False, mode=None, sample_rate=None):
        if sample_rate is None:
            raise ValueError("Sample rate = None")
        images = None
        self.current_normal_number = normal_number
        self.transform = transform
        org_images = [os.path.join(path, img) for img in os.listdir(path)]
        if mode == "train":
            images = random.sample(org_images, int(len(org_images)*sample_rate))
        elif mode == "test":
            images = org_images
        else:
            raise ValueError("WDNMD")
        # print("ORG SIZE -> {}, SAMPLED SIZE -> {}".format(len(org_images), len(images)) )
        images = sorted(images)
        self.images = images

    def __getitem__(self, index):
        image_path = self.images[index]
        # label = image_path.split('/')[-1].split('.')[0]
        label = image_path.split('/')[-2]
        # data = Image.open(image_path)
        data = default_loader(image_path)

        # data = TF.adjust_contrast(data, contrast_factor=1.5)
        data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.images)

def load_train(train_path, sample_rate):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    imagenet_data = MvtecDataLoader(train_path, transform=transform, mode="train", sample_rate=sample_rate)
    # imagenet_data = torchvision.datasets.ImageFolder(train_path, transform=transform)

    train_data_loader = torch.utils.data.DataLoader(imagenet_data,
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=num_worker,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=False)
    return train_data_loader, imagenet_data.__len__()


def init_c(DataLoader, net, eps=0.1):
    net.c = None
    c = torch.zeros((1, 64+128+256+256)).to('cuda')
    net.eval()
    n_samples = 0
    with torch.no_grad():
        for index, (images, label) in enumerate(DataLoader):
            # get the inputs of the batch
            img = images.to('cuda')
            outputs = net.encoder(img)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    return c

sig_f=1
def init_sigma(DataLoader, net):
    net.sigma = None
    net.eval()
    tmp_sigma = torch.tensor(0.0, dtype=torch.float).to('cuda')
    n_samples = 0
    with torch.no_grad():
        for index, (images, label) in enumerate(DataLoader):
            img = images.to('cuda')
            latent_z = net.encoder(img)
            diff = (latent_z - net.c) ** 2
            tmp = torch.sum(diff.detach(), dim=1)
            if (tmp.mean().detach() / sig_f) < 1:
                tmp_sigma += 1
            else:
                tmp_sigma += tmp.mean().detach() / sig_f
            n_samples += 1
    tmp_sigma /= n_samples
    return tmp_sigma

def train(args):

    train_dataset_loader, train_size = load_train(args.train_path, args.sample_rate)

    net = HS_Model()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0, 0.9), weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    net.c = None
    net.sigma = None
    net.c = init_c(train_dataset_loader, net)
    net.c.requires_grad = True
    net.sigma = init_sigma(train_dataset_loader, net)
    net.sigma.requires_grad = False

    with tqdm(range(len(train_dataset_loader))) as pbar:
        for i, (images, landmarks) in enumerate(tqdm(train_dataset_loader)):
            imgs = images.to(device=args.device)
            lams = landmarks.to(device=args.device)

            xrec, landmark_pred, latent_z = net(imgs)
            xrec_loss, landmark_loss = net.loss(imgs, xrec, lams, landmark_pred)
            compact_loss = (latent_z - net.c) ** 2
            loss_all = xrec_loss+1*landmark_loss+2*compact_loss
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            scheduler.step()

def get_parser():
    parser = argparse.ArgumentParser(description="HS-OCFER")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--sample_rate', type=float, default=1.0, help='sample ration')
    parser.add_argument('--train_path', type=str, default=r'./', help='Path to traindata')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train (default: 50)')
    parser.add_argument('--batch_size', default=12, type=int, help='Batch size per GPU')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--start_epoch', type=int, default=0, help='Number of epochs to train (default: 50)')

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    train(args)


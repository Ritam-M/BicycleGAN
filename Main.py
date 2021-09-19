import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from Model import *
from Dataset import *
from Train import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="edges2shoes", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--latent_dim", type=int, default=8, help="number of latent codes")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
parser.add_argument("--lambda_pixel", type=float, default=10, help="pixelwise loss weight")
parser.add_argument("--lambda_latent", type=float, default=0.5, help="latent loss weight")
parser.add_argument("--lambda_kl", type=float, default=0.01, help="kullback-leibler loss weight")
opt = parser.parse_args()
print(opt)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Loss functions
mae_loss = torch.nn.L1Loss()

# Initialize generator, encoder and discriminators
generator = Generator(opt.latent_dim, input_shape)
encoder = Encoder(opt.latent_dim, input_shape)
D_VAE = MultiDiscriminator(input_shape)
D_LR = MultiDiscriminator(input_shape)

if cuda:
    generator = generator.cuda()
    encoder.cuda()
    D_VAE = D_VAE.cuda()
    D_LR = D_LR.cuda()
    mae_loss.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
    encoder.load_state_dict(torch.load("saved_models/%s/encoder_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_VAE.load_state_dict(torch.load("saved_models/%s/D_VAE_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_LR.load_state_dict(torch.load("saved_models/%s/D_LR_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    D_VAE.apply(weights_init_normal)
    D_LR.apply(weights_init_normal)

# Optimizers
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_LR = torch.optim.Adam(D_LR.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, input_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
val_dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, input_shape, mode="val"),
    batch_size=8,
    shuffle=True,
    num_workers=1,
)

Train()

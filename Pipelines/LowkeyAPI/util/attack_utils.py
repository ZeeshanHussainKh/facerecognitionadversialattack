# Helper function for extracting features from pre-trained models
import os
import sys
import math
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
from PIL import Image
import numpy as np
from util.feature_extraction_utils import  warp_image, normalize_batch
from util.prepare_utils import get_ensemble, extract_features
import lpips
from tqdm import tqdm 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor_transform = transforms.ToTensor()
pil_transform = transforms.ToPILImage()






class Attack(nn.Module):

    def __init__(self, models, dim,attack_type, eps, c_sim=0.5, net_type='alex', lr=0.05,
                 n_iters=100, noise_size=0.001, n_starts=10, c_tv=None, sigma_gf=None, kernel_size_gf=None,
                 combination=False, warp=False, theta_warp=None, V_reduction=None):
        super(Attack, self).__init__()
        self.extractor_ens = get_ensemble(models, sigma_gf, kernel_size_gf, combination, V_reduction, warp, theta_warp)
        #print("There are '{}'' models in the attack ensemble".format(len(self.extractor_ens)))
        self.dim = dim
        self.eps = eps
        self.c_sim = c_sim
        self.net_type = net_type
        self.lr = lr
        self.n_iters = n_iters
        self.noise_size = noise_size
        self.n_starts = n_starts
        self.c_tv=None
        self.attack_type = attack_type
        self.warp = warp
        self.theta_warp = theta_warp
        if self.attack_type == 'lpips':
            self.lpips_loss = lpips.LPIPS(self.net_type).to(device)
    def execute(self, images, dir_vec, direction):

        images = Variable(images).to(device)
        dir_vec = dir_vec.to(device)
        # take norm wrt dim
        dir_vec_norm = dir_vec.norm(dim = 2).unsqueeze(2).to(device)
        dist = torch.zeros(images.shape[0]).to(device)
        adv_images = images.detach().clone()

        if self.warp:
            self.face_img = warp_image(images, self.theta_warp)

        for start in range(self.n_starts):
            # update adversarial images old and distance old
            adv_images_old = adv_images.detach().clone()
            dist_old = dist.clone()
            # add noise to initialize ( - noise_size, noise_size)
            noise_uniform = Variable(2 * self.noise_size * torch.rand(images.size()) - self.noise_size).to(device)
            adv_images = Variable(images.detach().clone() + noise_uniform, requires_grad=True).to(device)

            for i in range(self.n_iters):
                adv_features = extract_features(adv_images, self.extractor_ens, self.dim).to(device)
                # normalize feature vectors in ensembles 
                loss = direction*torch.mean((adv_features - dir_vec)**2/dir_vec_norm)

                if self.c_tv != None:
                    tv_out = self.total_var_reg(images, adv_images)
                    loss -= self.c_tv * tv_out

                if self.attack_type == 'lpips':
                    lpips_out = self.lpips_reg(images, adv_images)
                    loss -= self.c_sim * lpips_out

                grad = torch.autograd.grad(loss, [adv_images])
                adv_images = adv_images + self.lr * grad[0].sign()
                perturbation = adv_images - images

                if self.attack_type == 'sgd':
                    perturbation = torch.clamp(perturbation, min=-self.eps, max=self.eps)
                    adv_images = images + perturbation

            adv_images = torch.clamp(adv_images, min=0, max=1)
            adv_features = extract_features(adv_images, self.extractor_ens, self.dim).to(device)
            dist = torch.mean((adv_features - dir_vec)**2/dir_vec_norm, dim = [1,2])

            if direction ==1:
                adv_images[dist < dist_old] = adv_images_old[dist < dist_old]
                dist[dist < dist_old] = dist_old[dist < dist_old]
            else: 
                adv_images[dist > dist_old] = adv_images_old[dist > dist_old]
                dist[dist > dist_old] = dist_old[dist > dist_old]

        return adv_images.detach().cpu()


    def lpips_reg(self, images, adv_images):
        if self.warp:
            face_adv = warp_image(adv_images, self.theta_warp)
            lpips_out = self.lpips_loss(normalize_batch(self.face_img).to(device), normalize_batch(face_adv).to(device))[0][0][0][0] /(2*adv_images.shape[0])
            lpips_out += self.lpips_loss(normalize_batch(images).to(device), normalize_batch(adv_images).to(device))[0][0][0][0] / (2*adv_images.shape[0])

        else:
            lpips_out = self.lpips_loss(normalize_batch(images).to(device), normalize_batch(adv_images).to(device))[0][0][0][0] / adv_images.shape[0]

        return lpips_out

    def total_var_reg(images, adv_images):
        perturbation = adv_images - images
        tv = torch.mean(torch.abs(perturbation[:, :, :, :-1] - perturbation[:, :, :, 1:])) + \
         torch.mean(torch.abs(perturbation[:, :, :-1, :] - perturbation[:, :, 1:, :]))

        return tv





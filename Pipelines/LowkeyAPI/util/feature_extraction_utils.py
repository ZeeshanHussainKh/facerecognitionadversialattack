# Helper function for extracting features from pre-trained models
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def warp_image(tensor_img, theta_warp, crop_size=112):
    # applies affine transform theta to image and crops it 

    theta_warp = torch.Tensor(theta_warp).unsqueeze(0).to(device)
    grid = F.affine_grid(theta_warp, tensor_img.size())
    img_warped = F.grid_sample(tensor_img, grid)
    img_cropped = img_warped[:,:,0:crop_size, 0:crop_size]
    return(img_cropped)

def normalize_transforms(tfm, W,H):
    # normalizes affine transform from cv2 for pytorch
    tfm_t = np.concatenate((tfm, np.array([[0,0,1]])), axis = 0)
    transforms = np.linalg.inv(tfm_t)[0:2,:]
    transforms[0,0] = transforms[0,0]
    transforms[0,1] = transforms[0,1]*H/W
    transforms[0,2] = transforms[0,2]*2/W + transforms[0,0] + transforms[0,1] - 1

    transforms[1,0] = transforms[1,0]*W/H
    transforms[1,1] = transforms[1,1]
    transforms[1,2] = transforms[1,2]*2/H + transforms[1,0] + transforms[1,1] - 1

    return transforms

def l2_norm(input, axis = 1):
    # normalizes input with respect to second norm
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def de_preprocess(tensor):
    # normalize images from [-1,1] to [0,1]
    return tensor * 0.5 + 0.5

# normalize image to [-1,1]
normalize = transforms.Compose([
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

def normalize_batch(imgs_tensor):
    normalized_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        normalized_imgs[i] = normalize(img_ten)

    return normalized_imgs

def resize2d(img, size):
    # resizes image
    return (F.adaptive_avg_pool2d(img, size))



class face_extractor(nn.Module):
    def __init__(self, crop_size = 112, warp = False, theta_warp = None):
        super(face_extractor, self).__init__()
        self.crop_size = crop_size
        self.warp = warp
        self.theta_warp = theta_warp

    def forward(self, input):

        if self.warp:
            assert(input.shape[0] == 1)
            input = warp_image(input, self.theta_warp, self.crop_size)


        return input



class feature_extractor(nn.Module):
    def __init__(self, model, crop_size = 112, tta = True, warp = False, theta_warp = None):
        super(feature_extractor, self).__init__()
        self.model = model
        self.crop_size = crop_size
        self.tta = tta
        self.warp = warp
        self.theta_warp = theta_warp

        self.model = model


    def forward(self, input):

        if self.warp:
            assert(input.shape[0] == 1)
            input = warp_image(input, self.theta_warp, self.crop_size)


        batch_normalized = normalize_batch(input)
        batch_flipped = torch.flip(batch_normalized, [3])
        # extract features
        self.model.eval() # set to evaluation mode
        if self.tta:
            embed = self.model(batch_normalized) + self.model(batch_flipped)
            features = l2_norm(embed)
        else:
            features = l2_norm(self.model(batch_normalized))
        return features


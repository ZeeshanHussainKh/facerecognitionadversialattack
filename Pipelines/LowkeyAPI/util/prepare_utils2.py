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
import torchvision.datasets as datasets
import copy
import time
from util.feature_extraction_utils import feature_extractor, face_extractor, warp_image, de_preprocess, normalize_batch
import matplotlib.pyplot as plt
import lpips
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from tqdm import tqdm 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor_transform = transforms.ToTensor()
pil_transform = transforms.ToPILImage()



class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class MedianBlur(nn.Module):
    """
    Apply median blur filtering on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the median filter kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, dim=2):
        super(MedianBlur, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim

        self.kernel_size = kernel_size
        self.dim = dim

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )
        self.pad_size = int(kernel_size[0] / 2)

    def forward(self, input):
        """
        Apply median filter to input.
        Arguments:
            input (torch.Tensor): Input to apply median filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        input = F.pad(input, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), mode='reflect')
        filtered = input.clone()

        for c in range(filtered.shape[1]):
            if self.dim == 1:
                filtered[:, c, :] = torch.median(
                    input[:, c, :].unsqueeze(1).unfold(-1, self.kernel_size[0], 1), dim=-1
                ).values.squeeze(1)
            elif self.dim == 2:
                filtered[:, c, :, :] = torch.median(
                    input[:, c, :, :].unsqueeze(1).unfold(-2, self.kernel_size[0], 1)
                    .unfold(-1, self.kernel_size[1], 1), dim=-1
                ).values.squeeze(1)
            elif self.dim == 3:
                filtered[:, c, :, :, :] = torch.median(
                    input[:, c, :, :, :].unsqueeze(1).unfold(-3, self.kernel_size[0], 1)
                    .unfold(-2, self.kernel_size[1], 1).unfold(-1, self.kernel_size[2], 1), dim=-1
                ).values.squeeze(1)
            else:
                raise RuntimeError(
                    'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(self.dim)
                )

        return filtered



class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )
        self.pad_size = int(kernel_size[0] / 2)

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        input = F.pad(input, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), mode='reflect')
        return self.conv(input, weight=self.weight, groups=self.groups)




class dim_reduction(nn.Module):
    def __init__(self, V):
        super(dim_reduction, self).__init__()
        self.V = V

    def forward(self, input):
        return torch.matmul(input, self.V.to(input.device))



def get_ensemble_mb(models, sigma_gf, kernel_size_gf, combination, V_reduction, warp=False,
                                       theta_warp=None):
    # function prepares ensemble of feature extractors
    # outputs list of pytorch nn models 
    feature_extractor_ensemble = []
    if kernel_size_gf != None:
        # if apply gaussian filterng during attack
        gaussian_filtering = GaussianSmoothing(3, kernel_size_gf, sigma_gf)
        median_filtering = MedianBlur(3, kernel_size_gf)
        if V_reduction == None:
            for model in models:
                feature_extractor_model = nn.DataParallel(nn.Sequential(gaussian_filtering,median_filtering,
                                                                        feature_extractor(model=model, warp=warp,
                                                                                          theta_warp=theta_warp))).to(device)
                feature_extractor_ensemble.append(feature_extractor_model)
                if combination:
                    feature_extractor_model = nn.DataParallel(
                        feature_extractor(model=model, warp=warp, theta_warp=theta_warp)).to(device)
                    feature_extractor_ensemble.append(feature_extractor_model)

        else:
            for i, model in enumerate(models):
                feature_extractor_model = nn.DataParallel(
                    nn.Sequential(gaussian_filtering,median_filtering ,feature_extractor(model=model, warp=warp, theta_warp=theta_warp),
                                  dim_reduction(V_reduction[i]))).to(device)
                feature_extractor_ensemble.append(feature_extractor_model)
                if combination:
                    feature_extractor_model = nn.DataParallel(
                        nn.Sequential(feature_extractor(model=model, warp=warp, theta_warp=theta_warp),
                                      dim_reduction(V_reduction[i]))).to(device)
                    feature_extractor_ensemble.append(feature_extractor_model)

    else:
        if V_reduction == None:
            for model in models:
                feature_extractor_model = nn.DataParallel(
                    feature_extractor(model=model, warp=warp, theta_warp=theta_warp)).to(device)
                feature_extractor_ensemble.append(feature_extractor_model)
        else:
            for i, model in enumerate(models):
                feature_extractor_model = nn.DataParallel(
                    nn.Sequential(feature_extractor(model=model, warp=warp, theta_warp=theta_warp),
                                  dim_reduction(V_reduction[i]))).to(device)
                feature_extractor_ensemble.append(feature_extractor_model)

    return feature_extractor_ensemble


def extract_features(imgs, feature_extractor_ensemble, dim):
    # function computes mean feature vector of images with ensemble of feature extractors 

    features = torch.zeros(imgs.shape[0], len(feature_extractor_ensemble), dim)
    for i, feature_extractor_model in enumerate(feature_extractor_ensemble):
        # batch size, model in ensemble, dim
        features_model = feature_extractor_model(imgs)
        features[:, i, :] = features_model


    return features





def prepare_models(model_backbones,
                 input_size,
                 model_roots,
                 kernel_size_attack,
                 sigma_attack,
                 combination,
                 using_subspace,
                 V_reduction_root):

    backbone_dict = {'IR_50': IR_50(input_size), 'IR_152': IR_152(input_size), 'ResNet_50': ResNet_50(input_size),
                     'ResNet_152': ResNet_152(input_size)}

    print("Loading Attack Backbone Checkpoint '{}'".format(model_roots))
    print('=' * 20)

    models_attack = []
    for i in range(len(model_backbones)):
        model = backbone_dict[model_backbones[i]]
        model.load_state_dict(torch.load(model_roots[i]))
        models_attack.append(model)
        
    
    if using_subspace:

          V_reduction = []
          for i in range(len(model_backbones)):
            V_reduction.append(torch.tensor(np.load(V_reduction_root[i])))

            dim = V_reduction[0].shape[1]
    else:
            V_reduction = None
            dim = 512

    return models_attack, V_reduction, dim

def prepare_data(query_data_root, target_data_root, freq, batch_size, warp = False, theta_warp = None):

    data = datasets.ImageFolder(query_data_root, tensor_transform)

    subset_query = list(range(0, len(data), freq))
    subset_gallery = [x for x in list(range(0, len(data))) if x not in subset_query]
    query_set = torch.utils.data.Subset(data, subset_query)
    gallery_set = torch.utils.data.Subset(data, subset_gallery)

    if target_data_root != None:
        target_data =  datasets.ImageFolder(target_data_root, tensor_transform)
        target_loader = torch.utils.data.DataLoader(
            target_data, batch_size = batch_size)
    else:
        target_loader = None

    query_loader = torch.utils.data.DataLoader(
        query_set, batch_size = batch_size)
    gallery_loader = torch.utils.data.DataLoader(
        gallery_set, batch_size = batch_size)

    return query_loader, gallery_loader, target_loader


def prepare_dir_vec(dir_vec_extractor, imgs, dim, combination):
    dir_vec = extract_features(imgs, dir_vec_extractor, dim).detach().cpu()
    if combination:
        dir_vec = torch.repeat_interleave(dir_vec,2,1)
    return dir_vec 



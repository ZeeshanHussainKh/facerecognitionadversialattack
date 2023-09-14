import torch
from PIL import Image
import numpy as np
from util.feature_extraction_utils import feature_extractor, normalize_transforms, warp_image, normalize_batch
from backbone.model_irse import IR_50, IR_101, IR_152
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from util.attack_utils import  Attack
from util.prepare_utils import prepare_models, prepare_dir_vec, get_ensemble, prepare_data
from align.detector import detect_faces
from align.align_trans import get_reference_facial_points, warp_and_crop_face
import argparse
import matplotlib.pyplot as plt
import copy
import torchvision.transforms as transforms
import sys, os



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_tensor = transforms.ToTensor()

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

if __name__ == '__main__':



    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default = './', help = 'directory with images to protect')

    args = parser.parse_args()
    dir_root = args.dir


    eps = 0.05
    n_iters = 50
    input_size = [112, 112]
    attack_type = 'lpips'
    c_tv = None
    c_sim = 0.05
    lr = 0.0025
    net_type = 'alex'
    noise_size = 0.005
    n_starts = 1
    kernel_size_gf = 7
    sigma_gf = 3
    combination = True
    using_subspace = False
    V_reduction_root = './'
    model_backbones = ['IR_152', 'IR_152', 'ResNet_152', 'ResNet_152']
    model_roots = ['models/Backbone_IR_152_Arcface_Epoch_112.pth', 'models/Backbone_IR_152_Cosface_Epoch_70.pth', \
     'models/Backbone_ResNet_152_Arcface_Epoch_65.pth', 'models/Backbone_ResNet_152_Cosface_Epoch_68.pth'] 
    direction = 1
    crop_size = 112
    scale = crop_size / 112.




    models_attack, V_reduction, dim = prepare_models(model_backbones,
                 input_size,
                 model_roots,
                 kernel_size_gf,
                 sigma_gf,
                 combination,
                 using_subspace,
                 V_reduction_root)


    for img_name in os.listdir(dir_root):
        img_root = os.path.join(dir_root, img_name)
        print('Finding reference points')
        reference = get_reference_facial_points(default_square = True) * scale
        img = Image.open(img_root)
        h,w,c = np.array(img).shape

            # detect facial points
        _, landmarks = detect_faces(img)
        facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]

        # find transform
        _,tfm = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))

        # find pytorch transform
        theta = normalize_transforms(tfm, w, h)
        tensor_img = to_tensor(img).unsqueeze(0).to(device)

        V_reduction = None
        dim = 512

        # find direction vector
        dir_vec_extractor = get_ensemble(models = models_attack, sigma_gf = None, kernel_size_gf = None, combination = False, V_reduction = V_reduction, warp = True, theta_warp = theta)
        dir_vec = prepare_dir_vec(dir_vec_extractor, tensor_img, dim, combination)


        img_attacked = tensor_img.clone()
        attack = Attack(models_attack, dim, attack_type, eps, c_sim, net_type, lr,
            n_iters, noise_size, n_starts, c_tv, sigma_gf, kernel_size_gf,
            combination, warp=True, theta_warp=theta, V_reduction = V_reduction)

        img_attacked = attack.execute(tensor_img, dir_vec, direction).detach().cpu()


        img_attacked_pil = transforms.ToPILImage()(img_attacked[0])
        img_attacked_pil.save(img_root[:-4] + '_attacked.png')

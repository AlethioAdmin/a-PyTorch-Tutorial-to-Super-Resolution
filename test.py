import os
from os import path

import torch
from utils import *
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model checkpoints
srgan_checkpoint = "./checkpoint_srgan.pth.tar"
srresnet_checkpoint = "./checkpoint_srresnet.pth.tar"

# Load models
srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
srresnet.eval()
srgan_generator = torch.load(srgan_checkpoint)['generator'].to(device)
srgan_generator.eval()


def sr(img):
    # Load image, downsample to obtain low-res version
    lr_img = Image.open(img, mode="r").convert('RGB')

    # Super-resolution (SR) with SRResNet
    sr_img_srresnet = srresnet(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srresnet = sr_img_srresnet.squeeze(0).cpu().detach()
    sr_img_srresnet = convert_image(sr_img_srresnet, source='[-1, 1]', target='pil')

    # Super-resolution (SR) with SRGAN
    sr_img_srgan = srgan_generator(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srgan = sr_img_srgan.squeeze(0).cpu().detach()
    sr_img_srgan = convert_image(sr_img_srgan, source='[-1, 1]', target='pil')

    return sr_img_srresnet, sr_img_srgan


source_dir = '/home/shared/sr/input'
output_dir = '/home/shared/sr'


if __name__ == '__main__':
    os.makedirs(path.join(output_dir, 'srres'), exist_ok=True)
    os.makedirs(path.join(output_dir, 'srgan'), exist_ok=True)

    for entry in os.listdir(source_dir):
        if entry.endswith('.png'):
            input = path.join(source_dir, entry)
            sr_res, sr_gan = sr(input)
            sr_res.save(path.join(output_dir, 'srres', entry))
            sr_res.save(path.join(output_dir, 'srgan', entry))

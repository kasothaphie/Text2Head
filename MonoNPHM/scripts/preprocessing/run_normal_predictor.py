import os
import sys

import tyro
import numpy as np
from PIL import Image
from nphm_tum import env_paths

import torch
import torchvision.transforms.functional as ttf
from torchvision.transforms import InterpolationMode
sys.path.append(f"{env_paths.CODE_BASE}/src/nphm_tum/preprocessing/face_normals/resnet_unet/")
from nphm_tum.preprocessing.face_normals.resnet_unet.resnet_unet_model import ResNetUNet
from nphm_tum.utils.print_utils import print_flashy

NORMAL_MODEL_PATH = f"{env_paths.CODE_BASE}/src/nphm_tum/preprocessing/face_normals/pretrained_models/model.pth"


def get_face_bbox(lmks, img_size):
    """
    Computes facial bounding box as required in face_normals
    :param lmks:
    :param img_size:
    :return: (vertical_start, vertical_end, horizontal_start, horizontal_end)
    """

    #umin = np.min(lmks[:, 0]*img_size[1])
    #umax = np.max(lmks[:, 0]*img_size[1])
    #vmin = np.min((-0.0 + lmks[:, 1])*img_size[0])
    #vmax = np.max((-0.0 + lmks[:, 1])*img_size[0])

    umin = np.min(lmks[:, 0] )
    umax = np.max(lmks[:, 0] )
    vmin = np.min((-0.0 + lmks[:, 1]) )
    vmax = np.max((-0.0 + lmks[:, 1]) )

    umean = np.mean((umin, umax))
    vmean = np.mean((vmin, vmax))

    l = round(1.2 * np.max((umax - umin, vmax - vmin)))

    if l > np.min(img_size):
        l = np.min(img_size)

    us = round(np.max((0, umean - float(l) / 2)))
    ue = us + l

    vs = round(np.max((0, vmean - float(l) / 2)))
    ve = vs + l

    if ue > img_size[1]:
        ue = img_size[1]
        us = img_size[1] - l

    if ve > img_size[0]:
        ve = img_size[0]
        vs = img_size[0] - l

    us = int(us)
    ue = int(ue)

    vs = int(vs)
    ve = int(ve)

    return vs, ve, us, ue

def annotate_face_normals(img, lmks):
    model = ResNetUNet(n_class=3).cuda()
    model.load_state_dict(torch.load(NORMAL_MODEL_PATH))
    model.eval()


    img = ttf.to_tensor(img)
    img_size = img.shape[-2:]



    t, b, l, r = get_face_bbox(lmks, img_size)
    crop = img[:, t:b, l:r]
    crop = ttf.resize(crop, 256, InterpolationMode.BICUBIC)
    crop = crop.clamp(-1, 1) * 0.5 + 0.5

    # get normals out --> model returns tuple and normals are first element
    normals = model(crop[None].cuda())[0]

    # normalize normals
    normals = normals / torch.sqrt(torch.sum(normals ** 2, dim=1, keepdim=True))

    # rescale them to original resolution
    rescaled_normals = ttf.resize(
        normals[0], (b - t, r - l), InterpolationMode.BILINEAR
    )

    # create a normal image in sample['rgb'] resolution and add the rescaled normals at
    # the correct location
    masked_normals = torch.zeros_like(img)
    masked_normals[:, t:b, l:r] = rescaled_normals.cpu()

    # plot
    normal_img = ttf.to_pil_image(masked_normals * 0.5 + 0.5)
    return normal_img


def main(seq_name : str):
    print_flashy(f'[ENTERING - NORMAL PREDICTION] @ {seq_name}')


    files = [f for f in os.listdir(f'{env_paths.DATA_TRACKING}/{seq_name}/source') if f.endswith('.png')]
    files.sort()

    for i, f in enumerate(files):
        NORMAL_OUT_PATH = f'{env_paths.DATA_TRACKING}/{seq_name}/normals/'

        if os.path.exists(f'{NORMAL_OUT_PATH}/{f}'):
            continue
        img_path = f'{env_paths.DATA_TRACKING}/{seq_name}/source/{f}'
        lm_path = f'{env_paths.DATA_TRACKING}/{seq_name}/pipnet/test.npy'
        lm_path = f'{env_paths.DATA_TRACKING}/{seq_name}/kpt/{i:05d}.npy'

        os.makedirs(NORMAL_OUT_PATH, exist_ok=True)
        img = Image.open(img_path)
        lms = np.load(lm_path) #[0, ...]

        normal_img = annotate_face_normals(img, lms[:, :])
        #normal_img.show()
        normal_img.save(f'{NORMAL_OUT_PATH}/{f}')

    print_flashy(f'[EXITING - NORMAL PREDICTION] @ {seq_name}')


if __name__ == '__main__':
    tyro.cli(main)

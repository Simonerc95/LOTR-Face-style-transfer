#!/usr/bin/python
# -*- encoding: utf-8 -*-
from model import BiSeNet
import torch
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

import cv2

def vis_parsing_maps(im, parsing_anno, _resize, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]


    vis_parsing_anno = parsing_anno.copy()
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno = vis_parsing_anno.astype(dtype='uint8')


    # Save result or not
    if save_im:
        cv2.imwrite(save_path, cv2.resize(vis_parsing_anno, (_resize,_resize)))


def face_parsing(respth='./res/seg_res', dspth='./cropped', cp='79999_iter.pth', resize_size=256):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()


    with torch.no_grad():
        for image_path in os.listdir(dspth):
            print(image_path)
            print('data' , dspth)
            print('hereeeee',osp.join(dspth, image_path))
            img = Image.open(osp.join(dspth, image_path))
            img = transforms.ToTensor()(img)
            img = transforms.CenterCrop(min(img.shape[1:]))(img)
            img = transforms.Resize((512, 512))(img)
            image = transforms.ToPILImage()(img)
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            print(np.unique(parsing))
            if len(np.unique(parsing)) < 5:
                print(f"Warning: style transfer using only {len(np.unique(parsing))} masks from {image_path}, \n"
                      f"This may lead to bad results.")
            print('imageeeeeeeeeeee', image_path)

            vis_parsing_maps(image, parsing, resize_size, stride=1, save_im=True, save_path=osp.join(respth, image_path[:-4]+'.png'))





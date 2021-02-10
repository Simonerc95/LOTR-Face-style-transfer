import itertools as it
from operator import itemgetter

import networkx as nx
import nltk
import numpy as np
import tensorflow.compat.v1 as tf
from os.path import join

from components.PSPNet.model import load_color_label_dict
from components.path import WEIGHTS_DIR


def merge_segments(content_segmentation, style_segmentation):
    common_keys = set(content_segmentation.keys()).intersection(set(style_segmentation.keys()))

    if (5, 5, 5) in set(content_segmentation.keys()).difference(common_keys) and (4, 4, 4) in set(
            style_segmentation.keys):
        style_segmentation[(5, 5, 5)] = style_segmentation[(4, 4, 4)]
    if (4, 4, 4) in set(content_segmentation.keys()).difference(common_keys) and (5, 5, 5) in set(
            style_segmentation.keys):
        style_segmentation[(4, 4, 4)] = style_segmentation[(5, 5, 5)]

    # if one of the eyes was missing from the style image masks, but is present in the content, we replicate the only
    # present eye inside the style image as a virtual second eye to be transferred on the other eye of content image

    common_keys = list(set(content_segmentation.keys()).intersection(set(style_segmentation.keys())))
    new_content = {key: content_segmentation[key] for key in common_keys}
    new_style = {key: style_segmentation[key] for key in common_keys}

    assert new_content.keys() == new_style.keys()
    print("Segments merged")
    return new_content, new_style


def reduce_dict(dict, image):
    _, h, w, _ = image.shape
    arr = np.zeros((h, w, 3), int)
    for k, v in dict.items():
        I, J = np.where(v)
        arr[I, J] = k[::-1]
    return arr


def get_unique_colors_from_image(image):
    h, w, c = image.shape
    assert (c == 3)
    vec = np.reshape(image, (h * w, c))
    unique_colors = np.unique(vec, axis=0)
    return [tuple(color) for color in unique_colors]


def extract_segmentation_masks(segmentation, colors=None, flag=False, do_hair=False):
    if colors is None:
        # extract distinct colors from segmentation image
        colors = get_unique_colors_from_image(segmentation)
    if do_hair:
        if flag:  # used to keep the masks excluded from style transfer (True only for content image), in order to
            # apply them
            # to recover the corresponding pixels from the original image and placing them on the transferred image
            return {color: mask for (color, mask) in
                    ((color, np.all(segmentation.astype(np.int32) == color, axis=-1)) for color in colors if
                     color not in
                     [(0, 0, 0), (16, 16, 16)]) if  # (0,0,0) is background, (16,16,16) is body mask
                    mask.max()}, {color: mask for (color, mask) in
                                  ((color, np.all(segmentation.astype(np.int32) == color, axis=-1)) for color in
                                   [(0, 0, 0), (16, 16, 16), (17, 17, 17)]) if
                                  mask.max()}
        else:
            return {color: mask for (color, mask) in
                    ((color, np.all(segmentation.astype(np.int32) == color, axis=-1)) for color in colors if
                     color not in
                     [(0, 0, 0), (16, 16, 16)]) if
                    mask.max()}
    else:
        if flag:
            return {color: mask for (color, mask) in
                    ((color, np.all(segmentation.astype(np.int32) == color, axis=-1)) for color in colors if
                     color not in
                     [(0, 0, 0), (16, 16, 16), (17, 17, 17)]) if  # (0,0,0) is background, (16,16,16) is body mask,
                    # (17,17,17) is for hair
                    mask.max()}, {color: mask for (color, mask) in
                                  ((color, np.all(segmentation.astype(np.int32) == color, axis=-1)) for color in
                                   [(0, 0, 0), (16, 16, 16), (17, 17, 17)]) if
                                  mask.max()}
        else:
            return {color: mask for (color, mask) in
                    ((color, np.all(segmentation.astype(np.int32) == color, axis=-1)) for color in colors if
                     color not in
                     [(0, 0, 0), (16, 16, 16), (17, 17, 17)]) if
                    mask.max()}


def mask_for_tf(segmentation_mask):
    return [tf.expand_dims(tf.expand_dims(tf.constant(segmentation_mask[key].astype(np.float32)), 0), -1) for key
            in sorted(segmentation_mask)]

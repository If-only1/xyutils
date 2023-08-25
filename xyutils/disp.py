__all__ = ["color_dir", 'apply_colormap', 'get_colormap', 'show_depth_dir', 'show_depth', 'show_depth_pred_dir',
           "apply_colormap_relative"
           ]

import os
import logging
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm
from numba import jit

from .os import *


@jit
def get_disp_colormap():
    # 视差在256*32的colormap
    p_color_table = np.zeros(8192 * 3, dtype=np.uint8)
    p_color_table[0] = 0
    p_color_table[1] = 0
    p_color_table[2] = 0
    for i in range(1, 17):
        p_color_table[i * 3] = 255
        p_color_table[i * 3 + 1] = 255
        p_color_table[i * 3 + 2] = 255

    for i in range(17, 33):
        p_color_table[i * 3] = int(255 - (127.0 / (32 - 16)) * (i - 16))
        p_color_table[i * 3 + 1] = int(255 - (127.0 / (32 - 16)) * (i - 16))
        p_color_table[i * 3 + 2] = int(255 - (127.0 / (32 - 16)) * (i - 16))
    for i in range(33, 65):
        p_color_table[i * 3] = int(128 + (127.0 / (64 - 32)) * (i - 32))
        p_color_table[i * 3 + 1] = int(128 - (127.0 / (64 - 32)) * (i - 32))
        p_color_table[i * 3 + 2] = int(128 + (127.0 / (64 - 32)) * (i - 32))
    for i in range(65, 129):
        p_color_table[i * 3] = int(255 - (255.0 / (128 - 64)) * (i - 64))
        p_color_table[i * 3 + 1] = 0
        p_color_table[i * 3 + 2] = 255

    for i in range(129, 193):
        p_color_table[i * 3] = 0
        p_color_table[i * 3 + 1] = int((255.0 / (192 - 128)) * (i - 128))
        p_color_table[i * 3 + 2] = 255

    for i in range(193, 321):
        p_color_table[i * 3] = 0
        p_color_table[i * 3 + 1] = 255
        p_color_table[i * 3 + 2] = int(255 - (255.0 / (320 - 192)) * (i - 192))

    for i in range(321, 641):
        p_color_table[i * 3] = int((255.0 / (640 - 320)) * (i - 320))
        p_color_table[i * 3 + 1] = 255
        p_color_table[i * 3 + 2] = 0

    for i in range(641, 1281):
        p_color_table[i * 3] = 255
        p_color_table[i * 3 + 1] = int(255 - (255.0 / (1280 - 640)) * (i - 640))
        p_color_table[i * 3 + 2] = 0

    for i in range(1281, 8192):
        p_color_table[i * 3] = 255
        p_color_table[i * 3 + 1] = 0
        p_color_table[i * 3 + 2] = int((255.0 / (8192 - 1280)) * (i - 1280))

    return p_color_table


def get_depth_colormap():
    color = np.zeros((256, 10))
    for i in range(1, 256):
        color[i - 1:i] = i

    color = color.astype(np.uint8)
    color = cv2.applyColorMap(color, cv2.COLORMAP_HSV)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    p_color_table = np.zeros(8192 * 3, dtype=np.uint8)
    for i in range(128, ):
        for j in range(32):
            for k in range(3):
                p_color_table[i * 32 * 3 + j * 3 + k] = color[i * 2][0][k]
    for i in range(128, 256, ):
        for j in range(32):
            for k in range(3):
                p_color_table[i * 32 * 3 + j * 3 + k] = color[127 * 2][0][k]
    p_color_table[0] = 0
    p_color_table[1] = 0
    p_color_table[2] = 0
    return p_color_table


def get_res_colormap():
    color = np.zeros((256, 10))
    for i in range(1, 256):
        color[i - 1:i] = i

    color = color.astype(np.uint8)
    color = cv2.applyColorMap(color, cv2.COLORMAP_HSV)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    p_color_table = np.zeros(8192 * 3, dtype=np.uint8)
    for i in range(128 - 32, ):
        for j in range(32):
            for k in range(3):
                p_color_table[i * 32 * 3 + j * 3 + k] = color[(i + 32) * 2][0][k]
    for i in range(128 - 32, 256, ):
        for j in range(32):
            for k in range(3):
                p_color_table[i * 32 * 3 + j * 3 + k] = color[127 * 2][0][k]
    p_color_table[0] = 0
    p_color_table[1] = 0
    p_color_table[2] = 0
    return p_color_table


def get_colormap(type='disp'):
    if type == 'disp':
        return get_disp_colormap()
    elif type == 'depth':
        return get_depth_colormap()
    elif type == 'res':
        return get_res_colormap()
    else:
        raise ValueError


def apply_colormap(img, type='disp', scale=32, step=4):
    @jit
    def _coloring(_color_img):
        for _y in range(height):
            for x in range(width):
                for j in range(3):
                    _color_img[_y, x, j] = color_map[img[_y, x] * 3 + j]
        _color_img = _color_img
        for _y in range(0, min(8192, height * step), step):
            for j in range(3):
                _color_img[_y // step, width - 20:width, j] = color_map[_y * 3 + j]
        return _color_img

    height, width = img.shape
    img = img.astype(np.float32)
    img[img > 255] = 255
    img[img < 0] = 0

    img = img * scale
    img = img.astype(np.uint32)
    color_map = get_colormap(type)
    color_img = np.zeros((height, width, 3), dtype=np.uint8)
    color_img = _coloring(color_img)
    for y in range(0, min(8192, height * step), 300):
        cv2.putText(color_img, str(y // scale), (width - 20, y // step), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), )
    return color_img


def apply_colormap_relative(disp):
    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
    return disp_vis


def _process_one_img(img_name, pred_dir='./', out_dir='./', type='disp'):
    new_path = os.path.join(out_dir, img_name)
    new_path = add_new_ext(new_path, '.png')
    if os.path.isfile(new_path):
        logging.warning(f'{new_path} has been existed!')
        return
    img = cv2.imread(os.path.join(pred_dir, img_name), -1)
    if 'png' in img_name:
        img = img / 100
    new_img = apply_colormap(img, type)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(new_path, new_img)


def color_dir(pred_dir, max_nums=2000, out_dir=None, type='disp'):
    pred_dir = pred_dir + '/' if pred_dir[-1] != '/' else pred_dir
    if out_dir is None:
        out_dir = list(pred_dir.split('/'))
        out_dir[-3] = out_dir[-3] + '_vis'
        out_dir = '/'.join(out_dir)
        logging.info(f'color out dir {out_dir}')
    else:
        out_dir = out_dir + '/' if out_dir[-1] != '/' else out_dir
    mkdir(out_dir)
    p = Pool(8)  # 定义最大的进程数
    for f in tqdm(sorted(os.listdir(pred_dir))[:max_nums]):
        p.apply_async(_process_one_img, (f, pred_dir, out_dir, type), error_callback=logging.error)
    p.close()
    p.join()


def show_depth(left, depth, interval=1, add_number=True, bin_threshold=0, type='depth'):
    left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    color_map = get_colormap(type)
    if np.sum(depth != 0) > 100000 and interval == 1:
        return apply_colormap(depth, type=type)
    h, w = depth.shape
    i = 0
    for y in range(h):
        for x in range(w):
            i += 1
            if depth[y, x] != 0 and i % interval == 0:
                if not (0 <= depth[y, x] <= 255):
                    continue
                if bin_threshold > 0:
                    if depth[y, x] < bin_threshold:
                        left[y, x] = [255, 0, 0]
                    else:
                        left[y, x] = [0, 255, 0]
                else:
                    for j in range(3):
                        left[y, x, j] = color_map[int(depth[y, x] * 32) * 3 + j]
                if add_number and i % (100 * interval) == 0:
                    cv2.putText(left, f'{depth[y, x]:.1f}', (x, y), cv2.FONT_ITALIC, 0.5, (255, 0, 0))
    for _y in range(0, min(8192, h * 4), 4):
        for j in range(3):
            left[_y // 4, w - 20:w, j] = color_map[_y * 3 + j]
    for y in range(0, min(8192, h * 4), 300):
        cv2.putText(left, str(y // 32), (w - 20, y // 4), cv2.FONT_ITALIC, 0.5,
                    (255, 255, 255), )
    left = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
    return left


def show_depth_dir(data_root, left_name='left', depth_name='depth', save_name='depth_show', add_number=True,
                   bin_threshold=0, interval=10000):
    left_dir = os.path.join(data_root, left_name)
    fs = os.listdir(left_dir)
    save_dir = os.path.join(data_root, save_name)
    mkdir(save_dir)
    for f in tqdm(fs):
        left_p = os.path.join(left_dir, f)
        assert os.path.isfile(left_p)
        left = cv2.imread(left_p)
        depth_p = os.path.join(data_root, depth_name, f)
        depth_p = add_new_ext(depth_p, '.tiff')
        assert os.path.isfile(depth_p)
        depth = cv2.imread(depth_p, -1)
        left = show_depth(left, depth, add_number=add_number, bin_threshold=bin_threshold, interval=interval)
        save_p = os.path.join(save_dir, f)
        cv2.imwrite(save_p, left)


def show_depth_pred(left, depth, pred, interval=1, add_number=True):
    left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    color_map = get_colormap(type='depth')
    h, w = depth.shape
    i = 0
    for y in range(h):
        for x in range(w):
            i += 1
            if depth[y, x] != 0 and i % interval == 0 and pred[y, x] != 0:
                if not (0 <= depth[y, x] <= 255):
                    continue
                for j in range(3):
                    left[y, x, j] = color_map[int(depth[y, x] * 32) * 3 + j]
                if add_number and i % (100 * interval) == 0:
                    cv2.putText(left, f'{depth[y, x]:.1f}', (x, y), cv2.FONT_ITALIC, 0.5, (255, 0, 0))
                    cv2.putText(left, f'{pred[y, x]:.1f}', (x - 30, y), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
    for _y in range(0, min(8192, h * 4), 4):
        for j in range(3):
            left[_y // 4, w - 20:w, j] = color_map[_y * 3 + j]
    for y in range(0, min(8192, h * 4), 300):
        cv2.putText(left, str(y // 32), (w - 20, y // 4), cv2.FONT_ITALIC, 0.5,
                    (255, 255, 255), )
    left = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
    return left


def show_depth_pred_dir(data_root, left_name='left', depth_name='depth', pred_name='stereo_raft',
                        save_name='depth_show', add_number=True):
    left_dir = os.path.join(data_root, left_name)
    fs = os.listdir(left_dir)
    save_dir = os.path.join(data_root, save_name)
    mkdir(save_dir)
    for f in tqdm(fs):
        left_p = os.path.join(left_dir, f)
        assert os.path.isfile(left_p)
        left = cv2.imread(left_p)
        depth_p = os.path.join(data_root, depth_name, f)
        depth_p = add_new_ext(depth_p, '.tiff')
        assert os.path.isfile(depth_p)
        depth = cv2.imread(depth_p, -1)
        pred_p = os.path.join(data_root, pred_name, f)
        pred_p = add_new_ext(pred_p, '.tiff')
        assert os.path.isfile(pred_p)
        pred = cv2.imread(pred_p, -1)
        left = show_depth_pred(left, depth, pred, add_number=add_number)
        save_p = os.path.join(save_dir, f)
        cv2.imwrite(save_p, left)

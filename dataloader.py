from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List
import json
import pdb
import re
import io
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms.functional as tvf
import cv2

@dataclass
class Datum:
    img_path: os.PathLike | str
    bbox_path: os.PathLike | str
    calib_path: os.PathLike | str

@dataclass
class ImgCorners:
    img: np.ndarray | torch.Tensor
    corner: np.ndarray | torch.Tensor

    @staticmethod
    def collate(batch: List['ImgCorners']):
        imgs = torch.stack(
            [ic.img if isinstance(ic.img, torch.Tensor) else torch.from_numpy(ic.img) for ic in batch],
            dim=0
        )
        corners = torch.stack(
            [ic.corner if isinstance(ic.corner, torch.Tensor) else torch.from_numpy(ic.corner) for ic in batch],
            dim=0
        ).float()  # B,4,2
        return ImgCorners(imgs, corners)

class CornerSet(Dataset):
    def __init__(self, render_dir):
        self.data = []
        if not os.path.exists(render_dir):
            raise FileNotFoundError(f"{render_dir} not found!!")
        imgs_dir, bboxes_dir, calibs_dir = self._find_dirs(render_dir)
        for (img_path, bbox_path, calib_path) in zip(
            sorted([path for path in os.listdir(imgs_dir) if re.match(r"rgb_\d+", path)]),
            sorted([path for path in os.listdir(bboxes_dir) if re.match(r"bounding_box_3d_\d+", path)]),
            sorted([path for path in os.listdir(calibs_dir) if re.match(r"camera_params_\d+", path)])
        ):
            full_img = os.path.join(imgs_dir, img_path)
            if not self._is_valid_image(full_img):
                # skip corrupted image and its paired data
                continue
            self.data.append(
                Datum(
                    full_img,
                    os.path.join(bboxes_dir, bbox_path),
                    os.path.join(calibs_dir, calib_path)
                )
            )

    def _is_valid_image(self, path):
        try:
            with Image.open(path) as im:
                im.verify()  # verify header
            return True
        except Exception:
            return False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        try:
            img = torchvision.io.read_image(datum.img_path, mode=torchvision.io.ImageReadMode.RGB)
        except Exception:
            bgr = cv2.imread(datum.img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Failed to read image (corrupted): {datum.img_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

        bboxes = np.load(datum.bbox_path)
        # Take only first bbox (adjust if you need all)
        if len(bboxes) == 0:
            raise RuntimeError(f"No bboxes in {datum.bbox_path}")
        corners_h = self._find4corners(bboxes[0], 0.225, 0.120)[None, ...]  # 1,4,4

        with open(datum.calib_path, "r") as f:
            calib = json.load(f)

        w2c = np.array(calib['cameraViewTransform']).reshape(4, 4, order='F')
        projMat = np.array(calib['cameraProjection']).reshape(4, 4, order='F')
        corners_img_h = corners_h @ w2c.T @ projMat.T                # 1,4,4
        corners_img_ndc_h = corners_img_h[..., :-1] / corners_img_h[..., -1:]  # 1,4,3

        _, height, width = img.shape
        coords = torch.stack(
            (
                torch.from_numpy((corners_img_ndc_h[..., 0] + 1) / 2 * width),
                torch.from_numpy((1 - corners_img_ndc_h[..., 1]) / 2 * height),
            ),
            dim=-1
        ).float()  # 1,4,2
        coords = coords.squeeze(0)  # 4,2
        return ImgCorners(img, coords)

    def _find4corners(self, bbox_ele, box_width, box_height):
        corners = np.array(
            [
                [ bbox_ele['x_min'], bbox_ele['y_min'], bbox_ele['z_min'], 1.],
                [ bbox_ele['x_min'] + box_width, bbox_ele['y_min'], bbox_ele['z_min'], 1.],
                [ bbox_ele['x_min'] + box_width, bbox_ele['y_min'], bbox_ele['z_min'] + box_height, 1.],
                [ bbox_ele['x_min'], bbox_ele['y_min'], bbox_ele['z_min'] + box_height, 1.]
            ]
        )
        local2world = bbox_ele['transform']
        corners = corners @ local2world
        return corners

    def _find_dirs(self, render_dir: os.PathLike):
        imgs_dir = os.path.join(render_dir, "gripper_left_rgb")
        bboxes_dir = os.path.join(render_dir, "gripper_left_rgb_detection")
        calibs_dir = os.path.join(render_dir, "gripper_left_rgb_camera_calib")
        if not (os.path.exists(imgs_dir) and os.path.exists(bboxes_dir) and os.path.exists(calibs_dir)):
            raise FileNotFoundError(f"{imgs_dir} or {bboxes_dir} or {calibs_dir} doesn't exist!")
        return imgs_dir, bboxes_dir, calibs_dir




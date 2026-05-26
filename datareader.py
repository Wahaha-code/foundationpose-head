# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
import numpy as np


class Reader:
  def __init__(self, downscale=1, shorter_side=None):
    self.downscale = downscale
    self.K = np.loadtxt(f'cam_K_ob.txt').reshape(3,3)
    self.H = int(480)
    self.W = int(640)
    self.K[:2] *= self.downscale



  def get_mask(self, mask):
          # 假设传入的mask已经是一个numpy数组
          # 调整mask大小
          resized_mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
          # 将mask转换为bool类型，然后转为uint8
          final_mask = (resized_mask > 0).astype(np.uint8)
          return final_mask


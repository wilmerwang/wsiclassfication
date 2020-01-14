
import os
import sys
import numpy as np 
import openslide

class Patch(object):
    """
    采样获取补丁图像
    """
    def __init__(self, slide_path, mask, mask_level, patch_level):
        self.slide_path = slide_path
        self.mask = mask
        self.mask_level = mask_level
        self.patch_level = patch_level
        self.slide = openslide.OpenSlide(self.slide_path)
        self.factor = self.slide.level_downsamples[self.mask_level - self.patch_level]

    def sampled(self, number):
        """
        从全部掩码采样-->number数量的数据
        number: 采样数量
        return: 采样后的中心点，二维数组，比如[[11,30], [23, 34], [12, 34]]
        """
        X_idcs, Y_idcs = np.where(self.mask)
        vertices = np.round([X_idcs, Y_idcs]).transpose()* int(self.factor)

        if vertices.shape[0] < number:
            sampled_vertices = vertices
        else:
            sampled_vertices = vertices[np.random.randint(
                vertices.shape[0], size=number), :]

        return sampled_vertices

    def gen_patch(self, vertice, patch_size):
        """
        获取单个补丁
        vertice: [x_idc, y_idc]
        patch_size: 225, 256 等
        return: 一个以[x_idc, y_idc]为中心的，大小为patch_size的补丁图像
        """
        x = int(vertice[0] - patch_size/2)
        y = int(vertice[1] - patch_size/2)
        img = self.slide.read_region((x, y), 
            self.patch_level, 
            (patch_size, patch_size)).convert('RGB')

        return img




import os
import sys
import xml
import xml.etree.ElementTree as ET 
import numpy as np
import openslide
import cv2

class Mask(object):
    """
    获取whole slide images的掩码，比如去除空白背景后的tissue_mask、tumor_mask以及阳性WSI的normal_mask。
    """
    def __init__(self, slide_path, RGB_min, is_tumor, level):
        self.slide_path = slide_path
        self.RGB_min = RGB_min
        self.is_tumor = is_tumor
        self.level = level
        self.slide = openslide.OpenSlide(self.slide_path)

    def tissue_mask(self, output=None):
        """
        获取tissue mask, 去除空白背景.
        RGB_min: RGB通道最小值
        output: 如果包含output参数，将会输出一个掩码文件
        return: tissue mask
        """
        # 注意img_RGB的shape是原本slide.level_dimensions的转置
        img_RGB = np.transpose(np.array(
            self.slide.read_region((0, 0),self.level,self.slide.level_dimensions[self.level])\
                .convert("RGB")), axes=[1, 0, 2])

        img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)

        ret_R, th_R = cv2.threshold(img_RGB[:,:,0], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret_G, th_G = cv2.threshold(img_RGB[:,:,1], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret_B, th_B = cv2.threshold(img_RGB[:,:,2], 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret_S, th_S = cv2.threshold(img_HSV[:,:,1], 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        background_R = img_RGB[:, :, 0] > ret_R
        background_G = img_RGB[:, :, 1] > ret_G
        background_B = img_RGB[:, :, 2] > ret_B
        tissue_RGB = np.logical_not(background_R & background_G & background_B)
        tissue_S = img_HSV[:, :, 1] > ret_S
        min_R = img_RGB[:, :, 0] > self.RGB_min
        min_G = img_RGB[:, :, 1] > self.RGB_min
        min_B = img_RGB[:, :, 2] > self.RGB_min
        
        mask_tissue = tissue_S & tissue_RGB & min_R & min_G & min_B

        if output:
            np.save(output, mask_tissue)
        return mask_tissue

    def tumor_mask(self, in_xml, output=None):
        """
        利用xml注释获取tumor mask
        in_xml: 输入的xml注释
        output: tumor_mask掩码输出文件
        return： 返回tumor_mask
        """
        # 获取tumor区域边坐标
        self.in_xml = in_xml
        root = ET.parse(in_xml).getroot()
        annotations_tumor = root.findall('./Annotations/Annotation')
        tumor_polygons = []
        for annotation in annotations_tumor:
            X = list(map(lambda x: float(x.get('X')), 
                        annotation.findall('./Coordinates/Coordinate')))
            Y = list(map(lambda x: float(x.get('Y')), 
                        annotation.findall('./Coordinates/Coordinate')))
            vertices = np.round([X, Y]).astype(int).transpose().tolist()
            tumor_polygons.append(vertices)

        w, h = self.slide.level_dimensions[self.level]
        mask_tumor = np.zeros((h, w))
        factor = self.slide.level_downsamples[self.level]
        # 根据xml注释填充tumor区域
        for tumor_polygon in tumor_polygons:
            vertices = np.array(tumor_polygon) / factor
            vertices = vertices.astype(np.int32)
            cv2.fillPoly(mask_tumor, [vertices], (255))

        mask_tumor = mask_tumor[:] > 127
        mask_tumor = np.transpose(mask_tumor)

        if output:
            np.save(output, mask_tumor)
        return mask_tumor

    def normal_mask(self, in_xml, output=None):
        """
        获取正常区域掩码
        in_xml: 输入的xml注释
        output: normal_mask掩码输出文件
        return： 返回mask_normal
        """
        if self.is_tumor:
            mask_normal = self.tissue_mask() & (~ self.tumor_mask(in_xml))
        else:
            mask_normal = self.tissue_mask()

        if output:
            np.save(output, mask_normal)
        return mask_normal
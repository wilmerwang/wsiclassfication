# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET

import numpy as np
import openslide
import cv2


class Mask(object):
    """ Obtain the mask of WSIs.

    Args:
        slide_path (str): The path of whole slide image.
        rgb_min (int): The threshold value for removing background or WSI.
        level (int): WSI pyramid level from which to read file and make it's mask.
    """
    def __init__(self, slide_path, rgb_min, level):
        self.slide_path = slide_path
        self.rgb_min = rgb_min
        self.level = level
        self.slide = openslide.OpenSlide(self.slide_path)

    def tissue_mask(self):
        """Remove the background of WSI and obtain tissue mask.

        Return:
            The tissue mask of WSI
        """
        img_RGB = np.transpose(np.array(
            self.slide.read_region((0, 0),self.level,self.slide.level_dimensions[self.level])\
                .convert("RGB")), axes=[1, 0, 2])

        img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)

        ret_R, th_R = cv2.threshold(img_RGB[:, :, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret_G, th_G = cv2.threshold(img_RGB[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret_B, th_B = cv2.threshold(img_RGB[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret_S, th_S = cv2.threshold(img_HSV[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        background_R = img_RGB[:, :, 0] > ret_R
        background_G = img_RGB[:, :, 1] > ret_G
        background_B = img_RGB[:, :, 2] > ret_B
        tissue_RGB = np.logical_not(background_R & background_G & background_B)
        tissue_S = img_HSV[:, :, 1] > ret_S
        min_R = img_RGB[:, :, 0] > self.rgb_min
        min_G = img_RGB[:, :, 1] > self.rgb_min
        min_B = img_RGB[:, :, 2] > self.rgb_min
        
        mask_tissue = tissue_S & tissue_RGB & min_R & min_G & min_B
        return mask_tissue

    def tumor_mask(self, in_xml):
        """Obtain the tumor mask using .xml format annotation file.

        Args:
            in_xml (str): The path of .xml format annotation file.

        Return:
            The tumor mask.
        """
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

        for tumor_polygon in tumor_polygons:
            vertices = np.array(tumor_polygon) / factor
            vertices = vertices.astype(np.int32)
            cv2.fillPoly(mask_tumor, [vertices], 255)

        mask_tumor = mask_tumor[:] > 127
        mask_tumor = np.transpose(mask_tumor)
        return mask_tumor

    def normal_mask(self, in_xml):
        """Obtain normal mask from tumor WSI。
        Args:
            in_xml (str): The path to .xml format annotation file.

        Return:
            The normal mask.
        """
        mask_normal = self.tissue_mask() & (~ self.tumor_mask(in_xml))
        return mask_normal

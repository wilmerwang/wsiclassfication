# -*- coding: utf-8 -*-
import os

import openslide


class Patch(object):
    """Obtain the sampled patch image.

    Args:
        slide_path (str): The path to WSI files.
        patch_size (int): The size of sampled patch image.
        mask_level (int): WSI pyramid level from which to make the mask.
        patch_level (int): WSI pyramid level to sample WSI.
    """
    def __init__(self,
                 slide_path,
                 patch_size,
                 mask_level,
                 patch_level):
        self.slide_path = slide_path
        self.patch_size = patch_size
        self.mask_level = mask_level
        self.patch_level = patch_level

    def patch_gen(self, coords, output):
        """Save the patch using coords.

        Args:
            coords (2D array): The center vertices.
            output (str): The path to save patches.
        """
        slide = openslide.OpenSlide(self.slide_path)
        factor = slide.level_downsamples[self.mask_level - self.patch_level]
        i = 0
        for coord in coords:
            x = int(int(coord[0] * factor) - self.patch_size / 2)
            y = int(int(coord[1] * factor) - self.patch_size / 2)

            img = slide.read_region((x, y),
                                    self.patch_level,
                                    (self.patch_size, self.patch_size)).convert("RGB")

            slide_name = os.path.split(self.slide_path)[-1].spliet(".")[0]
            img.save(os.path.join(output, slide_name + str(i) + '.png'))
            i += 1

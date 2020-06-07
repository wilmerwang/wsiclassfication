# -*- coding: utf-8 -*-
import numpy as np


def random_sampled(mask, number):
    """Obtain the sampled vertices from WSI with RANDOM METHOD.

    Args:
        mask (str): The path to mask file.
        number (int): The number of sampled patches from mask.

    Return:
        The file with sampled vertices.
    """
    x_idcs, y_idcs = np.where(mask)
    vertices = np.round([x_idcs, y_idcs]).T

    if vertices.shape[0] < number:
        sampled_vertices = vertices
    else:
        sampled_vertices = vertices[np.random.randint(
            vertices.shape[0], size=number), :]

    return sampled_vertices

import numpy as np
import torch
from mmcv.transforms import BaseTransform, Compose

from embodiedscan.registry import TRANSFORMS
from embodiedscan.structures.points import get_points_type, DepthPoints
from lry_utils.utils_read import to_sample_idx, to_scene_id, NUM2RAW_3RSCAN
import os
import json




@TRANSFORMS.register_module()
class DefaultPipeline(BaseTransform):
    """Multiview data processing pipeline.

    The transform steps are as follows:

        1. Select frames.
        2. Re-ororganize the selected data structure.
        3. Apply transforms for each selected frame.
        4. Concatenate data to form a batch.

    Args:
        transforms (list[dict | callable]):
            The transforms to be applied to each select frame.
        n_images (int): Number of frames selected per scene.
        ordered (bool): Whether to put these frames in order.
            Defaults to False.
    """

    def __init__(self, ordered=False, keep_rgb=True):
        super().__init__()


    def transform(self, results: dict) -> dict:
        """Transform function.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: output dict after transformation.
        """
         # adding pcd info
        pc, color, _,_ = torch.load(results['pc_file'])
        
        points = np.concatenate([pc, color], axis=-1)
        points = DepthPoints(points,
                            points_dim=6,
                            attribute_dims=dict(color=[
                                points.shape[1] - 3,
                                points.shape[1] - 2,
                                points.shape[1] - 1,
                            ]))
        results['points'] = points
        return results

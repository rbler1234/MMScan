import os
import json
from pytorch3d.io import load_obj
import numpy as np
import torch


def process_trscan(scan_id, data_root, axis_align):
    axis_align_matrix = axis_align[scan_id]

    lidar_obj_path = os.path.join(data_root, scan_id, "mesh.refined.v2.obj")

    points, faces, aux = load_obj(lidar_obj_path)
    constant = torch.ones((points.shape[0], 1))
    points_extend = (torch.concat([points, constant], dim=-1)).numpy()

    uvs = aux.verts_uvs
    texture_images = aux.texture_images["mesh.refined_0"]
    texture_images = torch.flip(texture_images, dims=[0])

    uvs = uvs.unsqueeze(0)
    texture_images = texture_images.unsqueeze(0)

    pc_colors = (
        torch.nn.functional.grid_sample(
            texture_images.permute(0, 3, 1, 2),
            (uvs * 2 - 1).unsqueeze(0),
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(1)
        .permute(1, 0)
        .numpy()
    )

    points_trans = np.array(
        (points_extend @ axis_align_matrix.transpose())[:, :3]
    ).astype(np.float32)
    return points_trans, pc_colors, pc_colors[:, 0]

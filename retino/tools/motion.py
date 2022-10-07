"""Apply motion to fMRI data."""
import numpy as np
import scipy as sp

from scipy.ndimage import affine_transform


def param2mat_spm(p):
    """Create a Transform Matrix from SPM data."""
    T = np.eye(4)
    T[:3, 3] = p[0:3]

    Rx = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(p[3]), np.sin(p[3]), 0],
            [0, -np.sin(p[3]), np.cos(p[3]), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    Ry = np.array(
        [
            [np.cos(p[4]), 0, np.sin(p[4]), 0],
            [0, 1, 0, 0],
            [-np.sin(p[4]), 0, np.cos(p[4]), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    Rz = np.array(
        [
            [np.cos(p[5]), np.sin(p[5]), 0, 0],
            [-np.sin(p[5]), np.cos(p[5]), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    return T @ (Rx @ Ry @ Rz)


def apply_motion(image, motion):
    """Apply the motion to the image."""
    transf_mat = param2mat_spm(motion)
    output = affine_transform(
        image,
        np.linalg.inv(transf_mat),
        order=5,
        mode="nearest",
    )
    return output

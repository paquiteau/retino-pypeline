"""Apply motion to fMRI data."""
import numpy as np
import scipy as sp


def apply_motion(image, motion, center_point=None, reverse=False):
    """
    Apply a rigid motion to an image.

    Parameters
    ----------
    image: numpy.ndarray
        3D image to move.
    motion: numpy.ndarray
        1 x 6 array with the following order:
        tx, ty, tz, rx, ry, rz
    center_point: numpy.ndarray, default None
        Center point for the rotation, if None, the center of the
        image is considered.
    reverse: bool, default False
        if True, the motion is reversed.
    """
    if reverse:
        motion = -motion

    t_motion, angles = motion[:3], motion[3:]

    R = compute_rot_matrix(angles)
    corrected = apply_rotation(image, R, center_point=center_point)
    corrected = apply_translation(corrected, t_motion)
    return corrected


def apply_translation(image, t_motion):
    t_x, t_y, t_z = t_motion.astype(int)

    trans = np.copy(image)
    trans = np.roll(trans, t_x, axis=0)
    trans = np.roll(trans, t_y, axis=1)
    trans = np.roll(trans, t_z, axis=2)

    return trans


def compute_rot_matrix(angles):
    """Compute the rotation matrix from 3 angles in radians."""
    ax, ay, az = angles[0], angles[1], angles[2]

    Rx = np.array(
        [[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]]
    )
    Ry = np.array(
        [[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]]
    )
    Rz = np.array(
        [[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]]
    )
    return np.matmul(Rz, np.matmul(Ry, Rx))


def apply_rotation(image, rot_mat, center_point=None):
    """
    Apply rotation matrix to image with a specified center_point.

    Parameters
    ----------
    image: np.ndarray
    rot_mat: np.ndarray
    center_point: np.ndarray
    """
    if center_point is None:
        x_center, y_center, z_center = np.array(image.shape) // 2

    trans_mat_inv = np.linalg.inv(rot_mat)
    Nz, Ny, Nx = image.shape

    x = np.linspace(0, Nx - 1, Nx)
    y = np.linspace(0, Ny - 1, Ny)
    z = np.linspace(0, Nz - 1, Nz)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    coor = np.array([xx - x_center, yy - y_center, zz - z_center])

    coor_prime = np.tensordot(trans_mat_inv, coor, axes=((1), (0)))
    xx_prime = coor_prime[0] + x_center
    yy_prime = coor_prime[1] + y_center
    zz_prime = coor_prime[2] + z_center

    x_valid1 = xx_prime >= 0
    x_valid2 = xx_prime <= Nx - 1
    y_valid1 = yy_prime >= 0
    y_valid2 = yy_prime <= Ny - 1
    z_valid1 = zz_prime >= 0
    z_valid2 = zz_prime <= Nz - 1
    valid_voxel = x_valid1 * x_valid2 * y_valid1 * y_valid2 * z_valid1 * z_valid2
    z_valid_idx, y_valid_idx, x_valid_idx = np.where(valid_voxel > 0)

    image_transformed = np.zeros((Nz, Ny, Nx))

    data_w_coor = sp.interpolate.RegularGridInterpolator(
        (z, y, x), image, method="linear"
    )
    interp_points = np.array(
        [
            zz_prime[z_valid_idx, y_valid_idx, x_valid_idx],
            yy_prime[z_valid_idx, y_valid_idx, x_valid_idx],
            xx_prime[z_valid_idx, y_valid_idx, x_valid_idx],
        ]
    ).T
    interp_result = data_w_coor(interp_points)
    image_transformed[z_valid_idx, y_valid_idx, x_valid_idx] = interp_result

    return image_transformed

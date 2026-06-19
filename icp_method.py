"""
icp_method.py
传统 ICP（Iterative Closest Point）算法封装（WHU-TLS版参数）
"""

import time
import numpy as np
import open3d as o3d

def run_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_correspondence_distance: float = 0.05,
    max_iteration: int = 100,
    use_point_to_plane: bool = True,
    init_transform: np.ndarray = None
) -> dict:
    """
    运行传统 ICP 配准

    Parameters
    ----------
    source                : 源点云
    target                      : 目标点云
    max_correspondence_distance : 最大对应点搜索半径（归一化坐标）
    max_iteration               : 最大迭代次数
    use_point_to_plane          : True=点到面ICP，False=点到点ICP
    init_transform              : 初始变换矩阵，None=单位矩阵

    Returns
    -------
    dict : T_pred, time, fitness, inlier_rmse
    """
    if init_transform is None:
        init_transform = np.eye(4)

    if use_point_to_plane:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6,
        relative_rmse=1e-6,
        max_iteration=max_iteration
    )

    t_start = time.perf_counter()

    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance,
        init_transform,
        estimation,
        criteria
    )

    elapsed = time.perf_counter() - t_start

    return {
        'T_pred': np.array(result.transformation),
        'time'       : elapsed,
        'fitness'    : result.fitness,
        'inlier_rmse': result.inlier_rmse,
    }
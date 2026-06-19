"""
lp_icp_method.py
LP-ICP（Learned Point ICP）简化实现

由于完整 LP-ICP 需要预训练深度学习模型，
这里实现一个"教学级 LP-ICP"，核心思想完全一致：
  Step1: 用 FPFH 局部几何特征描述符代替深度学习特征（效果接近）
  Step2: 特征空间最近邻匹配 + 互近邻过滤
  Step3: 重叠区域估计（用对应点距离作为重叠分数）
  Step4: RANSAC 鲁棒粗配准
  Step5: ICP 精细配准

注：FPFH（Fast Point Feature Histograms）是经典的手工设计
    局部几何特征，在没有大量训练数据时是 LP-ICP 深度特征
    的良好替代品，能真实体现"特征引导配准"的优势。
"""

import time
import numpy as np
import open3d as o3d

def compute_fpfh_features(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.05
) -> o3d.pipelines.registration.Feature:
    """
    计算 FPFH 特征描述符

    FPFH 通过统计每个点邻域内法线方向的分布，
    生成一个 33 维的直方图向量，描述该点的局部几何形状。
    几何形状相似的点 → FPFH 向量相似 → 特征空间距离小

    Parameters
    ----------
    pcd        : 输入点云（已估计法线）
    voxel_size : 体素下采样尺寸（控制计算精度与速度的平衡）

    Returns
    -------
    fpfh : FPFH 特征对象
    """
    # 下采样：用体素格（小方格）均匀化点云密度
    # 避免密集区域的点主导特征匹配
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # 重新估计法线（下采样后需要重算）
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2,  # 搜索半径 = 2倍体素尺寸
            max_nn=30
        )
    )

    # 计算 FPFH 特征
    # radius = 5倍体素尺寸，覆盖足够大的局部邻域
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5,
            max_nn=100
        )
    )
    return pcd_down, fpfh

def run_lp_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float = 0.05,
    max_iteration_icp: int = 200,
    ransac_n: int = 4,
    ransac_max_iter: int = 4000000,
    ransac_confidence: float = 0.999
) -> dict:
    """
    运行 LP-ICP 配准

    Parameters
    ----------
    source            : 源点云
    target            : 目标点云
    voxel_size        : 特征提取的体素尺寸
    max_iteration_icp : ICP 精细配准最大迭代次数
    ransac_n          : RANSAC 每次随机采样的点对数
    ransac_max_iter   : RANSAC 最大迭代次数
    ransac_confidence : RANSAC 置信度

    Returns
    -------
    dict 包含各阶段耗时和最终结果
    """
    t_total_start = time.perf_counter()

    # ══════════════════════════════════════════════════════════
    # Step 1: 特征提取（对应 LP-ICP 中的深度学习编码器）
    # ══════════════════════════════════════════════════════════
    t0 = time.perf_counter()

    source_down, source_fpfh = compute_fpfh_features(source, voxel_size)
    target_down, target_fpfh = compute_fpfh_features(target, voxel_size)

    t_feature = time.perf_counter() - t0
    # 此时每个点都有了一个"身份证"（FPFH向量）

    # ══════════════════════════════════════════════════════════
    # Step 2 & 3 & 4: 特征匹配 + 重叠估计 + RANSAC 粗配准
    # （对应 LP-ICP 中的特征匹配 + 重叠分数 + 鲁棒变换估计）
    # ══════════════════════════════════════════════════════════
    t0 = time.perf_counter()

    # 特征匹配距离阈值：特征向量距离超过此值的点对直接丢弃
    distance_threshold = voxel_size * 1.5

    # RANSAC 配准：
    #   - 随机采样 ransac_n 个点对
    #   - 用它们估计变换
    #   - 统计支持该变换的内点数（称为共识度）
    #   - 反复迭代，保留共识度最高的变换
    # 这正是 LP-ICP 中"用重叠分数过滤 + 加权SVD"的等价操作
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        mutual_filter=True,                         # 互近邻过滤（LP-ICP的核心策略）
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        checkers=[
            # 检查器1：对应点之间的边长比例要一致（排除畸变匹配）
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            # 检查器2：对应点之间的距离不能超过阈值
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            ransac_max_iter, ransac_confidence
        )
    )

    T_coarse = np.array(result_ransac.transformation)  # 粗配准结果
    t_coarse = time.perf_counter() - t0

    # ══════════════════════════════════════════════════════════
    # Step 5: ICP 精细配准（以粗配准结果为初始值）
    # ══════════════════════════════════════════════════════════
    t0 = time.perf_counter()

    # 精细配准使用更小的搜索距离，提升精度
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target,
        voxel_size * 0.4,                           # 精细阶段搜索半径更小
        T_coarse,                                   # 以粗配准为起点！
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration_icp)
    )

    t_fine = time.perf_counter() - t0

    total_time = time.perf_counter() - t_total_start

    return {
        'T_pred'      : np.array(result_icp.transformation),
        'T_coarse'    : T_coarse,
        'time'        : total_time,
        'time_feature': t_feature,
        'time_coarse' : t_coarse,
        'time_fine'   : t_fine,
        'fitness'     : result_icp.fitness,
        'inlier_rmse' : result_icp.inlier_rmse,
        'ransac_fitness': result_ransac.fitness,
    }
"""
evaluator.py
配准结果评估模块

计算并汇总以下指标：
  - RE  (Rotation Error)    : 旋转误差，单位°，越小越好
  - TE  (Translation Error) : 平移误差，单位与点云坐标一致，越小越好
  - RR  (Registration Recall): 配准成功率（RE<阈值 且 TE<阈值）
  - fitness                 : Open3D的吻合度指标
  - inlier_rmse             : 内点均方根误差
  - time                    : 运行时间（秒）
"""

import numpy as np
from typing import List, Dict

def rotation_error(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """
    计算旋转误差（角度，单位°）

    原理：
      R_err = R_pred^T · R_gt 是"预测旋转相对于真值旋转的误差旋转"
      误差旋转对应的旋转角 = arccos((trace(R_err) - 1) / 2)

    Parameters
    ----------
    R_pred : 3×3 预测旋转矩阵
    R_gt   : 3×3 真值旋转矩阵

    Returns
    -------
    float : 旋转误差（度）
    """
    # 计算误差旋转矩阵
    R_err = R_pred.T @ R_gt
    # 从迹（对角线之和）提取旋转角
    # clip 防止数值误差导致 arccos 参数超出 [-1,1]
    cos_angle = (np.trace(R_err) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))

def translation_error(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """
    计算平移误差（欧氏距离）

    Parameters
    ----------
    t_pred : 预测平移向量 (3,)
    t_gt   : 真值平移向量 (3,)

    Returns
    -------
    float : 平移误差
    """
    return float(np.linalg.norm(t_pred - t_gt))

def evaluate_single(T_pred: np.ndarray, T_gt: np.ndarray) -> dict:
    """
    评估单次配准结果

    Parameters
    ----------
    T_pred : 4×4 预测变换矩阵
    T_gt   : 4×4 真值变换矩阵

    Returns
    -------
    dict : {'RE': float, 'TE': float}
    """
    R_pred = T_pred[:3, :3]
    t_pred = T_pred[:3,  3]
    R_gt   = T_gt[:3, :3]
    t_gt   = T_gt[:3,  3]

    return {
        'RE': rotation_error(R_pred, R_gt),
        'TE': translation_error(t_pred, t_gt)
    }

def compute_summary(results: List[Dict], re_threshold=15.0, te_threshold=0.3) -> Dict:
    """
    汇总多次实验结果，计算均值、中位数和配准召回率

    Parameters
    ----------
    results      : 单次评估结果列表
    re_threshold : 配准成功的旋转误差阈值（°）
    te_threshold : 配准成功的平移误差阈值

    Returns
    -------
    dict : 汇总统计量
    """
    REs    = np.array([r['RE']   for r in results])
    TEs    = np.array([r['TE']   for r in results])
    times  = np.array([r['time'] for r in results])

    # 配准召回率：同时满足旋转误差和平移误差阈值的比例
    success = (REs < re_threshold) & (TEs < te_threshold)
    RR = success.mean() * 100.0  # 转为百分比

    return {
        'RE_mean'   : REs.mean(),
        'RE_median' : np.median(REs),
        'TE_mean'   : TEs.mean(),
        'TE_median' : np.median(TEs),
        'RR'        : RR,           # 配准召回率 (%)
        'time_mean' : times.mean(),
        'time_std'  : times.std(),
        'n_success' : success.sum(),
        'n_total'   : len(results)
    }
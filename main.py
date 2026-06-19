"""
main.py
主实验程序：WHU-TLS 真实点云 ICP vs LP-ICP 全面对比实验

数据集：
  WHU-TLS（武汉大学地面激光扫描基准数据集）
  推荐场景：Mountain（山地）/ Forest（森林）↑ 低重叠+植被遮挡，最能体现 LP-ICP 的全局特征匹配优势

实验设计
--------实验A - 精度基准测试（相邻站配准）
  实验B - 鲁棒性：不同初始旋转扰动
  实验C - 鲁棒性：不同重叠率（点保留比例）
  实验D - 鲁棒性：不同测量噪声
  实验E - 速度：不同点云规模
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# ★★★ 用户配置区域（根据需要修改这里）★★★
# ══════════════════════════════════════════════════════════════

DATA_MODE = 'real'   # 固定为 'real'，使用 WHU-TLS 数据

# ── WHU-TLS 数据配置 ──────────────────────────────────────────
REAL_DATA_CONFIG = {
    'scene_dir': './data/WHU-TLS/3-Mountain',# 场景目录
    'scan_id_src' : 1,       # 源站编号
    'scan_id_tgt' : 2,       # 目标站编号（必须是已有真值对！）
    'use_raw'     : True,    # True=原始点云  False=已对齐点云
    'max_points'  : 20000,   # 每站最大加载点数
    'voxel_size'  : 0.05,    # 归一化坐标下体素尺寸
    'use_dong2018': False,   # 是否使用 Dong2018 版真值
}

# ── 全局实验参数 ───────────────────────────────────────────────
N_POINTS= 10000  # 仅合成模式使用，real模式由 max_points 控制
N_TRIALS= 5      # 重复次数（WHU-TLS加载慢，适当减少）

# WHU-TLS 坐标归一化后，阈值与合成数据不同：
# 归一化空间中点云尺度≈2，旋转误差阈值保持15°
# 平移误差阈值放宽到0.1（归一化坐标，对应实际约10~30m）
RE_THRESHOLD  = 15.0   # 旋转误差阈值（°）
TE_THRESHOLD  = 0.10   # 平移误差阈值（归一化坐标）

# ══════════════════════════════════════════════════════════════
# 模块导入
# ══════════════════════════════════════════════════════════════

from icp_method import run_icp
from lp_icp_method   import run_lp_icp
from evaluator       import evaluate_single, compute_summary
from visualizer      import (
    visualize_registration,
    plot_metric_vs_condition,
    plot_summary_bar,
    plot_speed_comparison,
    print_results_table,
)

# ══════════════════════════════════════════════════════════════
#统一数据接口（修改此函数以适配 WHU-TLS）
# ══════════════════════════════════════════════════════════════

def get_experiment_pair(
    overlap_ratio: float = 1.0,
    rotation_deg:float = 0.0,
    noise_std:     float = 0.0,
    seed:          int   = 42
):
    """
    统一数据接口（WHU-TLS 版）

    WHU-TLS 真实数据已自带真值变换，rotation_deg/noise_std
    作为"额外扰动"叠加在真值之上，用于实验B/C/D的受控测试。
    """

    from real_data_loader import create_real_experiment_pair
    return create_real_experiment_pair(
        overlap_ratio = overlap_ratio,
        rotation_deg  = rotation_deg,
        noise_std     = noise_std,
        seed          = seed,
        **REAL_DATA_CONFIG        # 透传场景路径、站编号等配置
    )

# ══════════════════════════════════════════════════════════════
# 单次实验核心函数（参数适配 WHU-TLS 尺度）
# ══════════════════════════════════════════════════════════════

def run_single_trial(
    overlap_ratio: float = 1.0,
    rotation_deg:  float = 0.0,
    noise_std:     float = 0.0,
    seed:          int   = 42,
    visualize:     bool  = False
) -> dict:
    """运行一次完整配准实验（ICP + LP-ICP）"""

    source, target, T_gt = get_experiment_pair(
        overlap_ratio = overlap_ratio,
        rotation_deg  = rotation_deg,
        noise_std     = noise_std,
        seed          = seed
    )

    # ── 传统ICP（参数适配 WHU-TLS 归一化坐标尺度）─────────────
    # 归一化后点云直径≈2，搜索半径用 0.05合适（原行星数据用0.1）
    icp_out = run_icp(
        source, target,
        max_correspondence_distance = 0.05,
        max_iteration= 100,
        use_point_to_plane          = True
    )

    # ── LP-ICP（FPFH+RANSAC 粗配准，适配真实 TLS 场景）──────────
    lpicp_out = run_lp_icp(
        source, target,
        voxel_size        = 0.05,   # 归一化坐标下的特征提取尺寸
        max_iteration_icp = 100
    )

    icp_eval= evaluate_single(icp_out['T_pred'],   T_gt)
    lpicp_eval = evaluate_single(lpicp_out['T_pred'], T_gt)

    icp_result = {
        **icp_eval,
        'time'   : icp_out['time'],
        'fitness': icp_out['fitness'],
        'rmse'   : icp_out['inlier_rmse']
    }
    lpicp_result = {
        **lpicp_eval,
        'time'   : lpicp_out['time'],
        'fitness': lpicp_out['fitness'],
        'rmse'   : lpicp_out['inlier_rmse']
    }

    if visualize:
        visualize_registration(source, target, icp_out['T_pred'],T_gt,title='ICP Registration Result')
        visualize_registration(source, target, lpicp_out['T_pred'], T_gt,
                               title='LP-ICP Registration Result')

    return {'icp': icp_result, 'lpicp': lpicp_result}

# ══════════════════════════════════════════════════════════════
# 实验 A：精度基准测试（以下实验函数体与原版完全相同，无需改动）
# ══════════════════════════════════════════════════════════════

def experiment_A_baseline():
    """
    实验A：基准精度测试

    使用 WHU-TLS 原始真值变换（无额外扰动），
    测试两种算法在真实数据上的配准精度下限。
    """
    print("\n" + "█" * 60)
    print("  Experiment A: Baseline Accuracy Test")
    print("  实验A：WHU-TLS 精度基准测试（真实真值变换）")
    print(f"  场景: {REAL_DATA_CONFIG['scene_dir'].split('/')[-1]} | "
          f"站{REAL_DATA_CONFIG['scan_id_src']}→站{REAL_DATA_CONFIG['scan_id_tgt']}")
    print("█" * 60)

    icp_results, lpicp_results = [], []

    for i in range(N_TRIALS):
        print(f"  Trial {i+1}/{N_TRIALS} ...", end='\r')
        out = run_single_trial(
            overlap_ratio = 1.0,
            rotation_deg  = 0.0,   # WHU-TLS 真实场景，不加额外旋转
            noise_std     = 0.0,
            seed          = i,
            visualize     = (i == 0)
        )
        icp_results.append(out['icp'])
        lpicp_results.append(out['lpicp'])

    print(f"  完成 {N_TRIALS} 次实验                ")

    summary_icp   = compute_summary(icp_results,   RE_THRESHOLD, TE_THRESHOLD)
    summary_lpicp = compute_summary(lpicp_results, RE_THRESHOLD, TE_THRESHOLD)

    print_results_table(summary_icp, summary_lpicp,
                        "Experiment A: WHU-TLS Baseline |实验A：WHU-TLS 基准精度")
    plot_summary_bar(summary_icp, summary_lpicp,save_path='result_A_baseline.png')
    return summary_icp, summary_lpicp

def experiment_B_rotation():
    """
    实验B：旋转鲁棒性测试

    在WHU-TLS 真值变换基础上叠加额外旋转扰动，
    模拟初始位姿估计误差不同时的配准表现。
    ICP 对大旋转几乎完全失败，LP-ICP 靠 RANSAC 粗配准保持鲁棒。
    """
    print("\n" + "█" * 60)
    print("  Experiment B: Rotation Robustness Test")
    print("  实验B：旋转扰动鲁棒性（WHU-TLS + 额外旋转）")
    print("█" * 60)

    # WHU-TLS 本身已有真实相对旋转，这里叠加额外扰动
    rotation_levels = [0, 5, 10, 20, 30, 45, 60, 90]
    icp_summaries, lpicp_summaries = [], []

    for rot in rotation_levels:
        icp_results, lpicp_results = [], []
        for i in range(N_TRIALS):
            out = run_single_trial(rotation_deg=rot, seed=i)
            icp_results.append(out['icp'])
            lpicp_results.append(out['lpicp'])

        s_icp   = compute_summary(icp_results,   RE_THRESHOLD, TE_THRESHOLD)
        s_lpicp = compute_summary(lpicp_results, RE_THRESHOLD, TE_THRESHOLD)
        icp_summaries.append(s_icp)
        lpicp_summaries.append(s_lpicp)
        print(f"  旋转扰动 {rot:3d}° → "
              f"ICP RR={s_icp['RR']:5.1f}%  LP-ICP RR={s_lpicp['RR']:5.1f}%")

    plot_metric_vs_condition(
        rotation_levels, icp_summaries, lpicp_summaries,
        x_label        = 'Extra Rotation Perturbation (°) |额外旋转扰动 (°)',
        condition_name = 'Rotation | 旋转',
        save_path      = 'result_B_rotation.png'
    )
    return icp_summaries, lpicp_summaries

def experiment_C_overlap():
    """
    实验C：低重叠率鲁棒性测试

    WHU-TLS 不同站之间本身重叠率就只有 40%~60%，
    再通过点保留比例进一步模拟低重叠场景。
    LP-ICP 的特征匹配对低重叠更鲁棒。
    """
    print("\n" + "█" * 60)
    print("  Experiment C: Overlap Ratio Robustness Test")
    print("  实验C：低重叠率鲁棒性（WHU-TLS 真实部分重叠）")
    print("█" * 60)

    overlap_levels = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3]
    icp_summaries, lpicp_summaries = [], []

    for ov in overlap_levels:
        icp_results, lpicp_results = [], []
        for i in range(N_TRIALS):
            out = run_single_trial(overlap_ratio=ov, seed=i)
            icp_results.append(out['icp'])
            lpicp_results.append(out['lpicp'])

        s_icp   = compute_summary(icp_results,   RE_THRESHOLD, TE_THRESHOLD)
        s_lpicp = compute_summary(lpicp_results, RE_THRESHOLD, TE_THRESHOLD)
        icp_summaries.append(s_icp)
        lpicp_summaries.append(s_lpicp)
        print(f"  重叠率 {ov:.0%} → "
              f"ICP RR={s_icp['RR']:5.1f}%  LP-ICP RR={s_lpicp['RR']:5.1f}%")

    plot_metric_vs_condition(
        overlap_levels, icp_summaries, lpicp_summaries,
        x_label        = 'Point Retention Ratio | 点保留比例',
        condition_name = 'Overlap | 重叠率',
        save_path      = 'result_C_overlap.png',
        invert_x = True
    )
    return icp_summaries, lpicp_summaries

def experiment_D_noise():
    """
    实验D：测量噪声鲁棒性测试

    TLS 实测噪声约3~5mm，这里在归一化坐标系下模拟
    不同强度的系统性噪声（如多径效应、植被穿透误差）。
    """
    print("\n" + "█" * 60)
    print("  Experiment D: Noise Robustness Test")
    print("  实验D：测量噪声鲁棒性（归一化坐标系）")
    print("█" * 60)

    # 归一化坐标系下，0.01对应实际约 1~3m的噪声（视场景尺度）
    noise_levels = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
    icp_summaries, lpicp_summaries = [], []

    for ns in noise_levels:
        icp_results, lpicp_results = [], []
        for i in range(N_TRIALS):
            out = run_single_trial(noise_std=ns, seed=i)
            icp_results.append(out['icp'])
            lpicp_results.append(out['lpicp'])

        s_icp   = compute_summary(icp_results,   RE_THRESHOLD, TE_THRESHOLD)
        s_lpicp = compute_summary(lpicp_results, RE_THRESHOLD, TE_THRESHOLD)
        icp_summaries.append(s_icp)
        lpicp_summaries.append(s_lpicp)
        print(f"  噪声σ={ns:.3f} → "
              f"ICP RR={s_icp['RR']:5.1f}%  LP-ICP RR={s_lpicp['RR']:5.1f}%")

    plot_metric_vs_condition(
        noise_levels, icp_summaries, lpicp_summaries,
        x_label        = 'Noise Std Dev σ | 噪声标准差',
        condition_name = 'Noise Level | 噪声强度',
        save_path      = 'result_D_noise.png'
    )
    return icp_summaries, lpicp_summaries

def experiment_E_speed():
    """
    实验E：速度测试（不同点数规模）

    通过调整 max_points 控制点云规模，
    测试两种算法的计算时间随点数的变化趋势。
    """
    print("\n" + "█" * 60)
    print("  Experiment E: Speed vs Point Cloud Size")
    print("  实验E：速度测试（不同点数规模）")
    print("█" * 60)

    point_sizes= [2000, 5000, 10000, 20000, 30000]
    icp_times, lpicp_times = [], []
    icp_stds,  lpicp_stds  = [], []

    for n in point_sizes:
        # 临时修改 max_pointsREAL_DATA_CONFIG['max_points'] = n
        t_icp_list, t_lpicp_list = [], []

        for i in range(N_TRIALS):
            out = run_single_trial(seed=i)
            t_icp_list.append(out['icp']['time'])
            t_lpicp_list.append(out['lpicp']['time'])

        icp_times.append(np.mean(t_icp_list))
        lpicp_times.append(np.mean(t_lpicp_list))
        icp_stds.append(np.std(t_icp_list))
        lpicp_stds.append(np.std(t_lpicp_list))
        print(f"  {n:6d} 点→ ICP {icp_times[-1]:.3f}s | "
              f"LP-ICP {lpicp_times[-1]:.3f}s")

    #恢复原始配置
    REAL_DATA_CONFIG['max_points'] = 30000

    plot_speed_comparison(
        point_sizes, icp_times, lpicp_times,
        icp_stds,    lpicp_stds,
        save_path = 'result_E_speed.png'
    )
    return icp_times, lpicp_times

# ══════════════════════════════════════════════════════════════
# 主程序入口
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "★" * 60)
    print("  WHU-TLS Point Cloud Registration")
    print("  WHU-TLS 真实点云配准实验：ICP vs LP-ICP")
    print("★" * 60)
    print(f"  Dataset| 数据集     : WHU-TLS")
    print(f"  Scene      | 场景       : {REAL_DATA_CONFIG['scene_dir'].split('/')[-1]}")
    print(f"  Scan pair  | 扫描站对: "
          f"{REAL_DATA_CONFIG['scan_id_src']} → {REAL_DATA_CONFIG['scan_id_tgt']}")
    print(f"  Max Points | 最大点数   : {REAL_DATA_CONFIG['max_points']:,}")
    print(f"  Voxel Size | 体素尺寸   : {REAL_DATA_CONFIG['voxel_size']}m")
    print(f"  Trials| 重复次数   : {N_TRIALS}")
    print(f"  RE Thresh  | 旋转阈值   : {RE_THRESHOLD}°")
    print(f"  TE Thresh  | 平移阈值   : {TE_THRESHOLD}")
    print()

    experiment_A_baseline()   # 精度基准
    experiment_B_rotation()   # 旋转鲁棒性
    experiment_C_overlap()    # 重叠率鲁棒性
    experiment_D_noise()      # 噪声鲁棒性
    experiment_E_speed()      # 速度对比

    print("\n" + "★" * 60)
    print("  All experiments done! | 全部实验完成！")
    print("  Output files | 输出文件：")
    print("    result_A_baseline.png")
    print("    result_B_rotation.png")
    print("    result_C_overlap.png")
    print("    result_D_noise.png")
    print("    result_E_speed.png")
    print("★" * 60 + "\n")
"""
real_data_loader.py
WHU-TLS 真实地面激光扫描点云数据加载器

目录结构（以3-Mountain 为例）：WHU-TLS/└── 3-Mountain/
      ├── 1-RawPointCloud/
      │   ├── 1.las ... 6.las
      ├── 2-AlignedPointCloud/
      │   ├── 1.las ... 6.las
      └── 3-GroundTruth/
          ├── 1-2/
          │   ├── transformation.txt
          │   └── transformation-Dong2018.txt
          ├── 2-3/
          ├── 3-4/
          ├── 5-4/
          └── 6-5/

说明：
  - 真值矩阵按"点云对"存放，文件夹名即为点云对编号（如 1-2 表示站1→站2）
  - 使用 1-RawPointCloud 作为点云源
  - transformation.txt 为本文件默认使用的真值
"""

import os
import numpy as np
import open3d as o3d
from typing import Tuple

#══════════════════════════════════════════════════════════════
# 工具：列出当前场景所有可用的点云对
# ══════════════════════════════════════════════════════════════

def list_available_pairs(scene_dir: str):
    """
    列出场景目录下所有可用的点云对

    Parameters
    ----------
    scene_dir : 场景目录，如 './data/WHU-TLS/3-Mountain'

    Returns
    -------
    pairs : list of (src_id, tgt_id) 元组，如 [(1,2),(2,3),(3,4),(5,4),(6,5)]
    """
    gt_dir = os.path.join(scene_dir, '3-GroundTruth')
    if not os.path.exists(gt_dir):
        print(f"[WHU-TLS] ⚠ 找不到 3-GroundTruth 目录：{gt_dir}")
        return []

    pairs = []
    for folder in sorted(os.listdir(gt_dir)):
        parts = folder.split('-')
        if len(parts) == 2and parts[0].isdigit() and parts[1].isdigit():
            pairs.append((int(parts[0]), int(parts[1])))

    print(f"[WHU-TLS] 场景 {os.path.basename(scene_dir)} 可用点云对：{pairs}")
    return pairs

# ══════════════════════════════════════════════════════════════
# 核心加载：单个 .las 文件 → numpy 点云
# ══════════════════════════════════════════════════════════════

def load_las_file(
    filepath:str,
    max_points: int = 30000,
    seed:       int = 0
) -> np.ndarray:
    """
    加载单个 .las / .laz 文件，返回 (N,3) 点云坐标（单位：米）

    Parameters
    ----------
    filepath   : .las 或 .laz 文件路径
    max_points : 随机下采样到此点数
    seed       : 随机种子

    Returns
    -------
    points : (N, 3) float64，单位：米
    """
    try:
        import laspy
    except ImportError:
        raise ImportError(
            "读取 .las 文件需要安装 laspy：\n"
            "  pip install laspy lazrs-python"
        )

    print(f"[WHU-TLS] 读取: {os.path.basename(filepath)}")

    with laspy.open(filepath) as f:
        las = f.read()

    # laspy 自动应用 scale + offset，直接得到物理米坐标
    points = np.stack([
        np.array(las.x, dtype=np.float64),
        np.array(las.y, dtype=np.float64),
        np.array(las.z, dtype=np.float64),
    ], axis=1)

    # 过滤 NaN / Inf
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    print(f"[WHU-TLS] 原始点数: {len(points):,}")

    # 随机下采样
    if len(points) > max_points:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(points), max_points, replace=False)
        points = points[idx]
        print(f"[WHU-TLS] 下采样后: {len(points):,} 点")

    return points

# ══════════════════════════════════════════════════════════════
# 核心加载：真值变换矩阵
# ══════════════════════════════════════════════════════════════

def load_gt_matrix(
    scene_dir:str,
    src_id:     int,
    tgt_id:     int,
    use_dong2018: bool = False
) -> np.ndarray:
    """
    加载两个扫描站之间的真值变换矩阵（4×4）

    WHU-TLS 真值目录结构：
      3-GroundTruth/{src_id}-{tgt_id}/
          transformation.txt          ← 默认使用
          transformation-Dong2018.txt ← 可选

    Parameters
    ----------
    scene_dir    : 场景根目录，如 './data/WHU-TLS/3-Mountain'
    src_id       : 源扫描站编号
    tgt_id       : 目标扫描站编号
    use_dong2018 : True 则使用 Dong2018 版真值，False 使用标准真值

    Returns
    -------
    T : (4, 4) float64 变换矩阵（源坐标系 → 目标坐标系）
    """
    fname = 'transformation-Dong2018.txt' if use_dong2018 else 'transformation.txt'
    gt_path = os.path.join(
        scene_dir, '3-GroundTruth',
        f'{src_id}-{tgt_id}',
        fname
    )

    if not os.path.exists(gt_path):
        #尝试反向（如目录是 tgt-src）
        gt_path_rev = os.path.join(
            scene_dir, '3-GroundTruth',
            f'{tgt_id}-{src_id}',
            fname
        )
        if os.path.exists(gt_path_rev):
            print(f"[WHU-TLS] 使用反向真值目录: {tgt_id}-{src_id}")
            T_rev = np.loadtxt(gt_path_rev, dtype=np.float64)
            assert T_rev.shape == (4, 4)
            # 反向取逆得到正向变换
            return np.linalg.inv(T_rev)
        else:
            available = list_available_pairs(scene_dir)
            raise FileNotFoundError(
                f"找不到真值文件：{gt_path}\n"
                f"当前场景可用点云对：{available}\n"
                f"请在 main.py 的REAL_DATA_CONFIG 中修改 scan_id_src / scan_id_tgt"
            )

    T = np.loadtxt(gt_path, dtype=np.float64)
    assert T.shape == (4, 4), f"真值矩阵形状错误: {T.shape}"
    print(f"[WHU-TLS] 真值矩阵已加载: {os.path.basename(os.path.dirname(gt_path))}/{fname}")
    return T

# ══════════════════════════════════════════════════════════════
# 工具：numpy 点云 → Open3D，体素下采样 + 法线估计
# ══════════════════════════════════════════════════════════════

def to_o3d(
    points:np.ndarray,
    color:      list= None,
    voxel_size: float = 0.05
) -> o3d.geometry.PointCloud:
    """
    numpy 点云转 Open3D 格式，估计法线（point-to-plane ICP 必需）

    Parameters
    ----------
    points     : (N, 3)归一化点云
    color      : [r, g, b]，None 表示不着色
    voxel_size : 体素尺寸（归一化坐标），0 表示不下采样

    Returns
    -------
    pcd : Open3D PointCloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if color is not None:
        pcd.paint_uniform_color(color)

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    radius = voxel_size * 5if voxel_size > 0 else 0.1
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(30)

    return pcd

# ══════════════════════════════════════════════════════════════
# 主接口：生成一对配准实验用的点云（供 main.py 调用）
# ══════════════════════════════════════════════════════════════

def create_real_experiment_pair(
    #── WHU-TLS 路径配置 ──────────────────────────────────────
    scene_dir:    str   = './data/WHU-TLS/3-Mountain',
    scan_id_src:  int   = 1,       # 源扫描站编号
    scan_id_tgt:  int   = 2,       # 目标扫描站编号
    use_raw:      bool  = True,    # True=用1-RawPointCloud，False=用 2-AlignedPointCloud
    max_points:   int   = 20000,
    voxel_size:   float = 0.05,    # 归一化坐标下的体素尺寸
    use_dong2018: bool  = False,   # 是否使用 Dong2018 版真值
    # ── 实验扰动（叠加在真值之上，用于实验B/C/D）─────────────
    overlap_ratio: float = 1.0,
    rotation_deg:  float = 0.0,
    noise_std:     float = 0.0,
    seed:          int   = 42,
    # ── 吸收 main.py 透传的多余参数 ──────────────────────────
    **kwargs
) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, np.ndarray]:
    """
    加载 WHU-TLS 两个扫描站点云+ 真值变换矩阵

    工作流程：
      1. 从 1-RawPointCloud 加载两站.las 点云
      2. 从 3-GroundTruth/{src}-{tgt}/transformation.txt 加载真值
      3. 联合归一化（质心对齐，尺度归一）
      4. 可选：叠加重叠裁剪 /旋转扰动 / 噪声（用于受控实验）
      5. 转为 Open3D 格式，估计法线

    Parameters
    ----------
    scene_dir    : 场景目录（如 './data/WHU-TLS/3-Mountain'）
    scan_id_src  : 源扫描站编号（1~6）
    scan_id_tgt  : 目标扫描站编号⚠ 必须是 3-GroundTruth 中存在的点云对！   可用对：(1,2),(2,3),(3,4),(5,4),(6,5)
    use_raw      : True=原始点云，False=已对齐点云
    max_points   : 每站最大加载点数
    voxel_size   : 归一化坐标下的体素尺寸
    use_dong2018 : 是否使用 Dong2018 版真值矩阵
    overlap_ratio: 点保留比例，<1 时裁剪模拟低重叠（实验C）
    rotation_deg : 额外旋转扰动°（实验B）
    noise_std    : 高斯噪声标准差（实验D）
    seed         : 随机种子

    Returns
    -------
    source_pcd : Open3D 点云（橙色）
    target_pcd : Open3D 点云（蓝色）
    T_gt: (4,4) 归一化坐标系下源→目标的真值变换矩阵
    """
    rng = np.random.RandomState(seed)

    # ── 1. 确定点云子目录 ─────────────────────────────────────
    pc_subdir = '1-RawPointCloud' if use_raw else '2-AlignedPointCloud'
    pc_dir = os.path.join(scene_dir, pc_subdir)

    src_path = os.path.join(pc_dir, f'{scan_id_src}.las')
    tgt_path = os.path.join(pc_dir, f'{scan_id_tgt}.las')

    #自动尝试 .laz 后缀
    for attr, path in [('src_path', src_path), ('tgt_path', tgt_path)]:
        if not os.path.exists(path):
            laz = path.replace('.las', '.laz')
            if os.path.exists(laz):
                if attr == 'src_path':
                    src_path = laz
                else:
                    tgt_path = laz
            else:
                available = list_available_pairs(scene_dir)
                raise FileNotFoundError(
                    f"找不到点云文件：{path}\n"
                    f"请检查 scene_dir 和 scan_id 是否正确。\n"
                    f"当前场景可用点云对：{available}"
                )

    # ── 2. 加载点云（.las → numpy）───────────────────────────
    src_pts = load_las_file(src_path, max_points, seed)
    tgt_pts = load_las_file(tgt_path, max_points, seed + 1)

    # ── 3. 加载真值变换矩阵 ──────────────────────────────────
    T_gt_raw = load_gt_matrix(scene_dir, scan_id_src, scan_id_tgt, use_dong2018)

    # ── 4. 联合归一化 ─────────────────────────────────────────
    all_pts= np.vstack([src_pts, tgt_pts])
    centroid = all_pts.mean(axis=0)
    bbox_diag = np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0))
    scale= bbox_diag / 2.0

    src_pts_n = (src_pts - centroid) / scale
    tgt_pts_n = (tgt_pts - centroid) / scale

    # 归一化坐标系下的真值矩阵（旋转不变，平移按比例缩放）
    T_gt = T_gt_raw.copy()
    T_gt[:3, 3] = T_gt_raw[:3, 3] / scale

    # 打印基本信息
    _cos = np.clip((np.trace(T_gt[:3, :3]) - 1) / 2, -1, 1)
    print(f"[WHU-TLS] 场景: {os.path.basename(scene_dir)} | "
          f"站{scan_id_src} → 站 {scan_id_tgt}")
    print(f"[WHU-TLS] 归一化尺度: {scale:.2f} m")
    print(f"[WHU-TLS] 真值旋转角: {np.degrees(np.arccos(_cos)):.2f}°  "
          f"真值平移(归一化): {np.linalg.norm(T_gt[:3, 3]):.4f}")

    # ── 5. 模拟低重叠（实验C）────────────────────────────────
    if overlap_ratio < 1.0:
        n_src = int(len(src_pts_n) * overlap_ratio)
        n_tgt = int(len(tgt_pts_n) * overlap_ratio)
        src_pts_n = src_pts_n[
            np.argsort(np.linalg.norm(src_pts_n, axis=1))[:n_src]]
        tgt_pts_n = tgt_pts_n[
            np.argsort(np.linalg.norm(tgt_pts_n, axis=1))[:n_tgt]]# ── 6. 叠加旋转扰动（实验B）─────────────────────────────
    if rotation_deg > 0:
        axis = rng.randn(3)
        axis /= np.linalg.norm(axis)
        angle = np.deg2rad(rotation_deg)
        K = np.array([
            [0,-axis[2],  axis[1]],
            [ axis[2],  0,       -axis[0]],
            [-axis[1],  axis[0],  0      ]
        ])
        R_perturb = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        t_perturb = rng.randn(3) * 0.01
        src_pts_n = (R_perturb @ src_pts_n.T).T + t_perturb
        T_perturb = np.eye(4)
        T_perturb[:3, :3] = R_perturb
        T_perturb[:3,3] = t_perturb
        T_gt = T_gt @ np.linalg.inv(T_perturb)

    # ── 7.叠加测量噪声（实验D）─────────────────────────────
    if noise_std > 0:
        src_pts_n += rng.randn(*src_pts_n.shape) * noise_std
        tgt_pts_n += rng.randn(*tgt_pts_n.shape) * noise_std

    # ── 8. 转为 Open3D 格式，估计法线 ────────────────────────
    source_pcd = to_o3d(src_pts_n, [1.0, 0.5, 0.0], voxel_size)
    target_pcd = to_o3d(tgt_pts_n, [0.0, 0.5, 1.0], voxel_size)

    print(f"[WHU-TLS] Open3D点数→ 源: {len(source_pcd.points):,}  "
          f"目标: {len(target_pcd.points):,}")

    return source_pcd, target_pcd, T_gt
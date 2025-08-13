import open3d as o3d
import numpy as np
import os
import glob
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings
import logging
from typing import Optional, Tuple, Dict, Any, List
import time
import csv
try:
    import pymeshfix  # type: ignore
    _HAS_PYMESHFIX = True
except Exception:
    _HAS_PYMESHFIX = False


class TqdmLoggingHandler(logging.Handler):
    """A logging handler that writes messages using tqdm.write to avoid breaking progress bars."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            pass


def setup_logger(log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("pointcloud_to_stl")
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    # Clear existing handlers
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = TqdmLoggingHandler()
    sh.setLevel(logger.level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logger.level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


logger = logging.getLogger("pointcloud_to_stl")


def pointcloud_to_stl(
    pointcloud_file_path,
    output_path,
    method='poisson',
    depth=10,
    width=0,
    scale=1.1,
    linear_fit=False,
    density_quantile=0.005,
    fill_holes=True,
    hole_ratio=0.02,
    remove_outliers=True,
    nb_neighbors=30,
    std_ratio=3.0,
    cluster_cleanup=False,
    cluster_eps_scale=1.5,
    cluster_min_points=50,
    planarize=False,
    planarize_strength=1.0,
    align_to_z=True,
    collect_stats: bool = False,
):
    """
    从点云文件重建STL网格并保存
    
    Args:
        pointcloud_file_path (str): 点云文件路径（.npy格式）
        output_path (str): 输出STL文件路径
        method (str): 重建方法，'poisson' 或 'ball_pivoting'
        depth (int): 八叉树深度，用于泊松重建；<=0 时自动估算
        width (float): 球旋转半径，用于球旋转重建
        scale (float): 泊松重建的尺度参数
        linear_fit (bool): 是否使用线性拟合
        density_quantile (float): 密度分位数阈值，用于移除噪声
        fill_holes (bool): 是否填充网格孔洞
    """
    start_time = time.perf_counter()
    stats: Dict[str, Any] = {
        "input": pointcloud_file_path,
        "output": output_path,
        "method": method,
        "depth": depth,
        "width_or_alpha": width,
        "scale": scale,
        "linear_fit": bool(linear_fit),
        "density_quantile": float(density_quantile) if density_quantile is not None else None,
        "hole_ratio": float(hole_ratio),
        "cluster_cleanup": bool(cluster_cleanup),
        "planarize": bool(planarize),
        "planarize_strength": float(planarize_strength),
    }
    try:
        # 始终使用CPU
        pass
        
        # 加载点云数据
        point_cloud_data = np.load(pointcloud_file_path)
        
        # 检查数据格式
        if point_cloud_data.shape[1] == 6:
            # 包含法向量的点云 (x, y, z, nx, ny, nz)
            points = point_cloud_data[:, :3]
            normals = point_cloud_data[:, 3:6]
        elif point_cloud_data.shape[1] == 3:
            # 只有坐标的点云 (x, y, z)
            points = point_cloud_data
            normals = None
        else:
            raise ValueError(f"不支持的点云格式，期望3或6列，实际{point_cloud_data.shape[1]}列")
        stats["num_input_points"] = int(points.shape[0])
        
        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 如果有法向量，直接使用；否则估算法向量
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        else:
            # 使用 Hybrid 半径+邻居 数的法向估计，并统一朝向与归一化
            try:
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3.0, max_nn=30))
            except Exception:
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
            try:
                pcd.orient_normals_consistent_tangent_plane(k=10)
            except Exception:
                pass
            pcd.normalize_normals()



        # 预处理：移除游离点 / 异常点
        if remove_outliers:
            try:
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
                logger.info(f"已移除统计离群点 (nb_neighbors={nb_neighbors}, std_ratio={std_ratio})")
            except Exception as e_rm:
                warnings.warn(f"统计离群点移除失败: {e_rm}")


        

        # 预处理：基于聚类保留最大连通簇（默认关闭）
        if cluster_cleanup and len(pcd.points) > 0:
            try:
                # 估计邻域尺度
                nn_dists = pcd.compute_nearest_neighbor_distance()
                mean_nn = float(np.mean(nn_dists)) if len(nn_dists) > 0 else 0.0
                eps = mean_nn * cluster_eps_scale if mean_nn > 0 else 0.005
                labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=max(5, int(cluster_min_points))))
                if labels.size > 0 and labels.max() >= 0:
                    largest_label = int(np.bincount(labels[labels >= 0]).argmax())
                    inliers = labels == largest_label
                    pcd = pcd.select_by_index(np.where(inliers)[0])
                    logger.info(f"已保留最大簇 (eps~{eps:.5f}, min_points={max(5, int(cluster_min_points))}) | 点数: {len(pcd.points)}")
            except Exception as e_clu:
                warnings.warn(f"DBSCAN聚类清理失败: {e_clu}")

        # 预处理：平面化（默认关闭）
        if planarize and len(pcd.points) > 0:
            try:
                pts_np = np.asarray(pcd.points)
                centroid = pts_np.mean(axis=0)
                # PCA法向
                centered = pts_np - centroid
                cov = centered.T @ centered / max(1, centered.shape[0] - 1)
                evals, evecs = np.linalg.eigh(cov)
                normal = evecs[:, np.argmin(evals)]
                normal = normal / (np.linalg.norm(normal) + 1e-12)
                # 投影到平面
                if planarize_strength > 0:
                    # 原始点到平面投影
                    distances = (centered @ normal)
                    proj = pts_np - np.outer(distances, normal)
                    s = float(np.clip(planarize_strength, 0.0, 1.0))
                    pts_np = (1.0 - s) * pts_np + s * proj
                # 可选：将平面法向对齐到+Z，便于生成近似平面网格
                if align_to_z:
                    target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                    v = np.cross(normal, target)
                    c = float(np.dot(normal, target))
                    if np.linalg.norm(v) > 1e-12:
                        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
                        R = np.eye(3) + vx + (vx @ vx) * (1.0 / (1.0 + c + 1e-12))
                        pts_np = (pts_np - centroid) @ R.T + centroid
                pcd.points = o3d.utility.Vector3dVector(pts_np)
                # 重新估算法向（Hybrid），统一朝向并归一化
                try:
                    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3.0, max_nn=30))
                except Exception:
                    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
                try:
                    pcd.orient_normals_consistent_tangent_plane(k=10)
                except Exception:
                    pass
                pcd.normalize_normals()
                logger.info("已执行平面化处理并对齐Z轴")
            except Exception as e_pl:
                warnings.warn(f"平面化处理失败: {e_pl}")

        # 根据选择的方法进行网格重建
        if method.lower() == 'poisson':
            mesh = reconstruct_poisson(
                pcd,
                depth,
                width,
                scale,
                linear_fit,
                density_quantile=density_quantile,
                fill_holes=fill_holes,
                hole_ratio=hole_ratio,
            )
        elif method.lower() == 'ball_pivoting':
            mesh = reconstruct_ball_pivoting(pcd, width)
        elif method.lower() == 'alpha_shape':
            # 将 width 解释为 alpha；若未提供则基于平均邻域估计
            mesh = reconstruct_alpha_shape(pcd, alpha=width)
        else:
            raise ValueError(f"不支持的重建方法: {method}")
        
        # 额外步骤：保留最大连通分量，去除孤立碎片
        try:
            labels = np.array(mesh.cluster_connected_triangles()[0])
            if labels.size > 0:
                largest_label = int(np.bincount(labels).argmax())
                tri_mask_remove = labels != largest_label
                mesh.remove_triangles_by_mask(tri_mask_remove)
                mesh.remove_unreferenced_vertices()
                logger.info("连通域: 保留最大连通分量")
        except Exception as e:
            warnings.warn(f"  连通域筛选失败: {e}")
        
        # 再次尝试补洞（有些情况下第一次补洞后仍可能存在小孔洞）
        if fill_holes:
            try:
                mesh = fill_mesh_holes_tensor(mesh, hole_ratio=hole_ratio)
                logger.info("二次补洞: 成功使用 tensor.fill_holes(hole_size=..) 并回写 legacy")
            except Exception as e:
                warnings.warn(f"  二次补洞: 失败，原因: {e}")

        # 平滑处理，减少噪声和锯齿（优先使用非收缩 Taubin）
        try:
            smooth_method = None
            if hasattr(mesh, "filter_smooth_taubin"):
                mesh = mesh.filter_smooth_taubin(number_of_iterations=50)
                smooth_method = "filter_smooth_taubin(iter=50)"
            else:
                mesh = mesh.filter_smooth_simple(number_of_iterations=10)
                smooth_method = "filter_smooth_simple(iter=10)"
            # 平滑后需要重新计算法向
            try:
                mesh.compute_vertex_normals()
            except Exception:
                pass
            logger.info(f"平滑: 成功使用 {smooth_method}")
        except AttributeError:
            # 旧版本 Open3D 无此API，忽略
            logger.info("平滑: Open3D 版本不支持平滑API，跳过")
        except Exception as e:
            warnings.warn(f"  平滑: 失败，原因: {e}")
        
        # 保存为STL文件
        success = o3d.io.write_triangle_mesh(output_path, mesh)
        
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        stats["elapsed_ms"] = elapsed_ms
        
        if success:
            logger.info(f"成功重建: {pointcloud_file_path} -> {output_path}")
            logger.info(f"点云点数: {len(points)} | 重建网格: 顶点 {len(mesh.vertices)} 面 {len(mesh.triangles)} | 用时 {elapsed_ms} ms")
            stats.update({
                "success": True,
                "num_vertices": int(len(mesh.vertices)),
                "num_triangles": int(len(mesh.triangles)),
            })
            if collect_stats:
                return True, stats
            return True
        else:
            logger.error(f"保存STL文件失败: {output_path}")
            stats.update({"success": False, "error": "write_triangle_mesh failed"})
            if collect_stats:
                return False, stats
            return False
            
    except Exception as e:
        logger.error(f"处理文件 {pointcloud_file_path} 时出错: {str(e)}")
        stats.update({"success": False, "error": str(e)})
        if collect_stats:
            return False, stats
        return False


def reconstruct_poisson(pcd, depth=10, width=0, scale=1.1, linear_fit=False,
                        density_quantile=0.005, fill_holes=True, hole_ratio: float = 0.02):
    """
    使用泊松重建方法重建网格
    
    Args:
        pcd: Open3D点云对象
        depth: 八叉树深度
        width: 宽度参数
        scale: 尺度参数
        linear_fit: 是否使用线性拟合
        density_quantile: 密度分位数阈值
        fill_holes: 是否填充孔洞
        device: 计算设备
    
    Returns:
        Open3D网格对象
    """
    # 确保深度为有效值；取消自动估算，默认使用10
    if depth is None or depth <= 0:
        depth = 10

    logger.info(f"使用泊松重建 (深度={depth}, 尺度={scale})")

    # 执行泊松重建
    # 注意：Open3D的泊松重建在此仅使用CPU
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit
    )
    
    # 计算法向量（保存STL需要）
    mesh.compute_vertex_normals()
    
    # 根据密度移除噪声顶点（密度阈值用于去除重建的杂乱部分）
    if density_quantile is not None and 0 < density_quantile < 1:
        threshold = np.quantile(densities, density_quantile)
        mesh.remove_vertices_by_mask(densities < threshold)

    # 网格清理
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.remove_non_manifold_edges()

    if fill_holes:
        try:
            mesh = fill_mesh_holes_tensor(mesh, hole_ratio=hole_ratio)
            logger.info("首次补洞: 成功使用 tensor.fill_holes(hole_size=..) 并回写 legacy")
        except Exception as e:
            warnings.warn(f"  首次补洞: 失败，原因: {e}")

    return mesh


def reconstruct_ball_pivoting(pcd, radius):
    """
    使用球旋转重建方法重建网格
    
    Args:
        pcd: Open3D点云对象
        radius: 球旋转半径
        device: 计算设备
    
    Returns:
        Open3D网格对象
    """
    logger.info(f"使用球旋转重建 (半径={radius})")
    
    # 如果未指定半径，自动估算
    if radius <= 0:
        # 计算点云的平均距离作为半径
        distances = pcd.compute_nearest_neighbor_distance()
        radius = np.mean(distances) * 2
    
    # 执行球旋转重建
    # 注意：球旋转重建目前主要在CPU上实现
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([radius, radius * 2, radius * 4])
    )
    
    # 计算法向量（保存STL需要）
    mesh.compute_vertex_normals()
    
    return mesh


def reconstruct_alpha_shape(pcd, alpha: float = 0.0):
    """
    使用 Alpha Shape 从（近）平面点云重建网格。

    当点云接近平面时，Alpha Shape 通常比泊松/球旋转更稳定。

    Args:
        pcd: Open3D 点云对象
        alpha: Alpha 参数；<=0 时自动估算（基于平均最近邻距离）

    Returns:
        Open3D 网格对象
    """
    if alpha is None or alpha <= 0:
        try:
            dists = pcd.compute_nearest_neighbor_distance()
            alpha = float(np.mean(dists)) * 2.0 if len(dists) > 0 else 0.01
            logger.info(f"自动估算 alpha: {alpha:.6f}")
        except Exception:
            alpha = 0.01
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    try:
        mesh.compute_vertex_normals()
    except Exception:
        pass
    # 清理
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh


def fill_mesh_holes_tensor(mesh: o3d.geometry.TriangleMesh, hole_ratio: float = 0.02,
                           enable_meshfix: bool = True) -> o3d.geometry.TriangleMesh:
    """
    使用张量API按给定比例估计 hole_size 并填充孔洞。

    流程：
    - 清理重复与无引用元素
    - 非流形检测，若允许且可用则用 pymeshfix 修复
    - 计算包围盒尺度估计 hole_size
    - 转 tensor mesh，执行 fill_holes(hole_size=..)，再转回 legacy
    - 重算法向
    """
    if mesh is None:
        return mesh

    # 预清理
    try:
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_triangles()
    except Exception:
        pass

    # 非流形修复（可选）
    try:
        is_edge_manifold = mesh.is_edge_manifold()
        is_vertex_manifold = mesh.is_vertex_manifold()
    except Exception:
        is_edge_manifold = True
        is_vertex_manifold = True

    if enable_meshfix and _HAS_PYMESHFIX and not (is_edge_manifold and is_vertex_manifold):
        try:
            vertices_np = np.asarray(mesh.vertices)
            triangles_np = np.asarray(mesh.triangles)
            fixer = pymeshfix.MeshFix(vertices_np, triangles_np)
            fixer.repair()
            repaired_vertices = fixer.mesh.vertices
            repaired_faces = fixer.mesh.faces
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(repaired_vertices),
                o3d.utility.Vector3iVector(repaired_faces),
            )
        except Exception as e:
            warnings.warn(f"pymeshfix 修复失败，继续执行填洞: {e}")

    # 估算 hole_size
    try:
        bbox_extent = mesh.get_max_bound() - mesh.get_min_bound()
        bbox_extent = np.asarray(bbox_extent, dtype=float)
        bbox_norm = float(np.linalg.norm(bbox_extent))
        hole_size = bbox_norm * float(hole_ratio)
        if not np.isfinite(hole_size) or hole_size <= 0:
            hole_size = max(1e-6, bbox_norm * 0.01)
    except Exception:
        hole_size = 1e-3

    # 张量填洞
    try:
        mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh).to(o3d.core.Device("CPU:0"))
        mesh_filled_tensor = mesh_tensor.fill_holes(hole_size=hole_size)
        mesh_filled = mesh_filled_tensor.to_legacy()
        try:
            mesh_filled.compute_vertex_normals()
        except Exception:
            pass
        return mesh_filled
    except Exception as e:
        # 失败则原样返回
        warnings.warn(f"tensor.fill_holes 失败，返回原网格: {e}")
        return mesh

def process_directory(input_dir, output_dir, method='poisson', 
                     depth=10, width=0, scale=1.1, linear_fit=False, device='CPU:0',
                     hole_ratio: float = 0.02, density_quantile: float = 0.005, 
                     remove_outliers: bool = True, nb_neighbors: int = 30, std_ratio: float = 3.0,
                     cluster_cleanup: bool = False, planarize: bool = False, planarize_strength: float = 1.0,
                     summary_csv: Optional[str] = None) -> None:
    """
    处理目录中的所有点云文件
    
    Args:
        input_dir (str): 输入目录路径
        output_dir (str): 输出目录路径
        method (str): 重建方法
        depth (int): 八叉树深度
        width (float): 球旋转半径
        scale (float): 泊松重建尺度参数
        linear_fit (bool): 是否使用线性拟合
        device (str): 计算设备
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    if summary_csv is None:
        summary_csv = os.path.join(output_dir, "reconstruction_summary.csv")
    
    # 查找所有.npy点云文件
    pointcloud_files: List[str] = []
    for file in os.listdir(input_dir):
        if file.lower().endswith('.npy'):
            pointcloud_files.append(os.path.join(input_dir, file))
    
    if not pointcloud_files:
        logger.warning(f"在目录 {input_dir} 中未找到.npy文件")
        return
    
    logger.info(f"找到 {len(pointcloud_files)} 个点云文件")
    
    # 处理每个文件
    success_count = 0
    rows: List[Dict[str, Any]] = []
    for pointcloud_file in tqdm(pointcloud_files, desc="处理点云文件"):
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(pointcloud_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}.stl")
        
        # 重建STL（收集统计信息）
        success, stats = pointcloud_to_stl(pointcloud_file, output_file, method, depth, width, scale, linear_fit,
                              density_quantile=density_quantile, fill_holes=True, hole_ratio=hole_ratio,
                              remove_outliers=remove_outliers, nb_neighbors=nb_neighbors, std_ratio=std_ratio,
                              cluster_cleanup=cluster_cleanup, planarize=planarize, planarize_strength=planarize_strength,
                              collect_stats=True)
        rows.append(stats)
        if success:
            success_count += 1
    
    # 写入汇总 CSV
    try:
        header = [
            "input", "output", "success", "error",
            "num_input_points", "num_vertices", "num_triangles", "elapsed_ms",
            "method", "depth", "width_or_alpha", "scale", "linear_fit",
            "density_quantile", "hole_ratio", "cluster_cleanup", "planarize", "planarize_strength",
        ]
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, None) for k in header})
        logger.info(f"汇总已写入: {summary_csv}")
    except Exception as e:
        warnings.warn(f"写入汇总CSV失败: {e}")
    
    logger.info(f"处理完成: {success_count}/{len(pointcloud_files)} 个文件成功重建")


def main():
    parser = argparse.ArgumentParser(description='从点云重建STL文件')
    parser.add_argument('input', nargs='?', default='output/pointcloud/insoles',
                       help='输入点云文件路径或目录 (默认: output/pointcloud/insoles)')
    parser.add_argument('output', nargs='?', default='output/raw/insoles',
                       help='输出STL文件路径或目录 (默认: output/raw/insoles)')
    parser.add_argument('--method', choices=['poisson', 'ball_pivoting', 'alpha_shape'], 
                       default='poisson', help='重建方法 (默认: poisson)')
    parser.add_argument('--depth', type=int, default=10, 
                       help='泊松重建的八叉树深度 (默认: 10)')
    parser.add_argument('--width', type=float, default=0, 
                       help='球旋转半径 (默认: 自动估算) 或 alpha_shape 的 alpha 值')
    parser.add_argument('--scale', type=float, default=1.1, 
                       help='泊松重建尺度参数 (默认: 1.1)')
    parser.add_argument('--linear-fit', action='store_true', 
                       help='使用线性拟合 (泊松重建)')
    parser.add_argument('--density-quantile', type=float, default=0.005, help='密度分位阈值（越小越完整，默认 0.005）')
    parser.add_argument('--hole-ratio', type=float, default=0.02, help='按包围盒尺度的孔洞填充比例 (默认 0.02)')
    # 仅保留离群点移除为默认流程；聚类清理和平面化默认关闭
    parser.add_argument('--no-outlier-removal', action='store_true', help='禁用统计离群点移除')
    parser.add_argument('--outlier-nb-neighbors', type=int, default=30, help='统计离群点移除的邻居数')
    parser.add_argument('--outlier-std-ratio', type=float, default=3.0, help='统计离群点移除的标准差阈值')
    parser.add_argument('--cluster-cleanup', action='store_true', help='启用基于聚类的游离点清理（默认关闭）')
    parser.add_argument('--planarize', action='store_true', help='启用平面化处理（默认关闭）')
    parser.add_argument('--planarize-strength', type=float, default=1.0, help='平面化强度[0,1]，1表示完全投影到平面')
    # 兼容下方对 args.device 的引用（目前重建过程仅使用CPU，此参数仅为兼容接口）
    parser.add_argument('--device', type=str, default='cpu', help='计算设备（当前未使用，仅为兼容接口）')
    # 日志与汇总
    parser.add_argument('--log-file', type=str, default=None, help='日志文件路径（可选，若提供会同时写入文件）')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG','INFO','WARNING','ERROR'], help='日志级别')
    parser.add_argument('--summary-csv', type=str, default=None, help='目录模式下的汇总CSV路径（默认写入到输出目录）')
    
    args = parser.parse_args()

    # 初始化日志
    setup_logger(args.log_file, args.log_level)
    
    # 检查输入路径
    if not os.path.exists(args.input):
        logger.error(f"输入路径不存在: {args.input}")
        return
    
    # 处理单个文件或目录
    if os.path.isfile(args.input):
        if not args.input.lower().endswith('.npy'):
            logger.error("输入文件必须是.npy格式")
            return
        
        # 确保输出目录存在
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        pointcloud_to_stl(
            args.input,
            args.output,
            args.method,
            args.depth,
            args.width,
            args.scale,
            args.linear_fit,
            density_quantile=args.density_quantile,
            fill_holes=True,
            hole_ratio=args.hole_ratio,
            remove_outliers=not args.no_outlier_removal,
            nb_neighbors=args.outlier_nb_neighbors,
            std_ratio=args.outlier_std_ratio,
            cluster_cleanup=args.cluster_cleanup,
            planarize=args.planarize,
            planarize_strength=args.planarize_strength,
        )
    
    elif os.path.isdir(args.input):
        process_directory(
            args.input,
            args.output,
            args.method,
            args.depth,
            args.width,
            args.scale,
            args.linear_fit,
            device=args.device,
            hole_ratio=args.hole_ratio,
            density_quantile=args.density_quantile,
            remove_outliers=not args.no_outlier_removal,
            nb_neighbors=args.outlier_nb_neighbors,
            std_ratio=args.outlier_std_ratio,
            cluster_cleanup=args.cluster_cleanup,
            planarize=args.planarize,
            planarize_strength=args.planarize_strength,
            summary_csv=args.summary_csv,
        )
    
    else:
        logger.error(f"无效的输入路径: {args.input}")


if __name__ == "__main__":
    main()

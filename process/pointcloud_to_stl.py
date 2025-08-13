import open3d as o3d
import numpy as np
import os
import glob
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings
try:
    import pymeshfix  # type: ignore
    _HAS_PYMESHFIX = True
except Exception:
    _HAS_PYMESHFIX = False


def pointcloud_to_stl(
    pointcloud_file_path,
    output_path,
    method='poisson',
    depth=10,
    width=0,
    scale=1.1,
    linear_fit=False,
    density_quantile=0.01,
    fill_holes=True,
    remove_outliers=True,
    nb_neighbors=20,
    std_ratio=2.0,
    every_k_points=None,
    voxel_size=None,
    recompute_normals=True,
    orient_k=50,
    cluster_cleanup=False,
    cluster_eps_scale=1.5,
    cluster_min_points=50,
    planarize=False,
    planarize_strength=1.0,
    align_to_z=True,
    smooth_method='taubin',
    smooth_iterations=50,
    decimate_target_triangles=0,
):
    """
    从点云文件重建STL网格并保存
    
    Args:
        pointcloud_file_path (str): 点云文件路径（.npy格式）
        output_path (str): 输出STL文件路径
        method (str): 重建方法，'poisson' / 'ball_pivoting' / 'alpha_shape'
        depth (int): 八叉树深度，用于泊松重建；建议 8-10 以避免过拟合噪声
        width (float): 球旋转/AlphaShape 参数（见具体方法）
        scale (float): 泊松重建的尺度参数
        linear_fit (bool): 是否使用线性拟合
        density_quantile (float): 泊松重建后，按密度分位数阈值移除低密度区域
        fill_holes (bool): 是否填充网格孔洞
        remove_outliers (bool): 是否进行统计离群点移除
        nb_neighbors (int): 统计离群点移除 - 邻居数量
        std_ratio (float): 统计离群点移除 - 标准差阈值
        every_k_points (int|None): 均匀下采样的 k，None 或 <=1 时禁用
        voxel_size (float|None): 体素下采样体素尺寸，None 或 <=0 时禁用
        recompute_normals (bool): 预处理后是否重新估计/统一法向
        orient_k (int): 法向一致性邻域大小（建议 50）
        cluster_cleanup (bool): 基于 DBSCAN 保留最大簇
        planarize (bool): 可选将点云投影到 PCA 主平面，并可对齐 Z 轴
        smooth_method (str): 网格平滑方法：'none'|'simple'|'laplacian'|'taubin'
        smooth_iterations (int): 平滑迭代次数
        decimate_target_triangles (int): 四元误差网格简化目标三角形数，<=0 禁用
    """
    try:
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
        
        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 如果有法向量，直接使用
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)

        # 下采样（可选）
        try:
            if voxel_size is not None and float(voxel_size) > 0:
                pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))
                print(f"  体素下采样: voxel_size={float(voxel_size)} | 点数 -> {len(pcd.points)}")
        except Exception as e:
            warnings.warn(f"体素下采样失败: {e}")
        try:
            if every_k_points is not None and int(every_k_points) > 1:
                pcd = pcd.uniform_down_sample(every_k_points=int(every_k_points))
                print(f"  均匀下采样: every_k_points={int(every_k_points)} | 点数 -> {len(pcd.points)}")
        except Exception as e:
            warnings.warn(f"均匀下采样失败: {e}")

        # 预处理：移除游离点 / 异常点
        if remove_outliers and len(pcd.points) > 0:
            try:
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio))
                print(f"  已移除统计离群点 (nb_neighbors={nb_neighbors}, std_ratio={std_ratio}) | 点数 -> {len(pcd.points)}")
            except Exception as e_rm:
                warnings.warn(f"统计离群点移除失败: {e_rm}")

        # 预处理：基于聚类保留最大连通簇（默认关闭）
        if cluster_cleanup and len(pcd.points) > 0:
            try:
                # 估计邻域尺度
                nn_dists = pcd.compute_nearest_neighbor_distance()
                mean_nn = float(np.mean(nn_dists)) if len(nn_dists) > 0 else 0.0
                eps = mean_nn * float(cluster_eps_scale) if mean_nn > 0 else 0.005
                labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=max(5, int(cluster_min_points))))
                if labels.size > 0 and labels.max() >= 0:
                    largest_label = int(np.bincount(labels[labels >= 0]).argmax())
                    inliers = labels == largest_label
                    pcd = pcd.select_by_index(np.where(inliers)[0])
                    print(f"  已保留最大簇 (eps~{eps:.5f}, min_points={max(5, int(cluster_min_points))}) | 点数: {len(pcd.points)}")
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
                    distances = (centered @ normal)
                    proj = pts_np - np.outer(distances, normal)
                    s = float(np.clip(planarize_strength, 0.0, 1.0))
                    pts_np = (1.0 - s) * pts_np + s * proj
                # 可选：将平面法向对齐到+Z
                if align_to_z:
                    target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                    v = np.cross(normal, target)
                    c = float(np.dot(normal, target))
                    if np.linalg.norm(v) > 1e-12:
                        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
                        R = np.eye(3) + vx + (vx @ vx) * (1.0 / (1.0 + c + 1e-12))
                        pts_np = (pts_np - centroid) @ R.T + centroid
                pcd.points = o3d.utility.Vector3dVector(pts_np)
                print("  已执行平面化处理并对齐Z轴" if align_to_z else "  已执行平面化处理")
            except Exception as e_pl:
                warnings.warn(f"平面化处理失败: {e_pl}")

        # 法向估计与一致性朝向（建议总是进行，以“平滑法向”）
        if recompute_normals and len(pcd.points) > 0:
            try:
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3.0, max_nn=30))
            except Exception:
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
            try:
                pcd.orient_normals_consistent_tangent_plane(k=int(orient_k))
            except Exception:
                pass
            try:
                pcd.normalize_normals()
            except Exception:
                pass

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
            )
        elif method.lower() == 'ball_pivoting':
            mesh = reconstruct_ball_pivoting(pcd, width)
        elif method.lower() == 'alpha_shape':
            mesh = reconstruct_alpha_shape(pcd, alpha=width)
        else:
            raise ValueError(f"不支持的重建方法: {method}")

        # 可选：再次补洞（某些情况下可能仍存在小孔）
        if fill_holes:
            try:
                mesh = fill_mesh_holes_tensor(mesh)
                print("  二次补洞: 成功使用 tensor.fill_holes(hole_size=..) 并回写 legacy")
            except Exception as e:
                warnings.warn(f"  二次补洞: 失败，原因: {e}")

        # 可选：网格简化（重采样）
        if isinstance(decimate_target_triangles, (int, float)) and int(decimate_target_triangles) > 0:
            try:
                mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(decimate_target_triangles))
                try:
                    mesh.compute_vertex_normals()
                except Exception:
                    pass
                print(f"  网格简化: 目标三角形数={int(decimate_target_triangles)}")
            except Exception as e:
                warnings.warn(f"网格简化失败: {e}")

        # 平滑处理（可选）
        try:
            sm = (smooth_method or 'none').lower()
            if sm == 'simple':
                mesh = mesh.filter_smooth_simple(number_of_iterations=int(smooth_iterations))
                print(f"  平滑: simple(iter={smooth_iterations})")
            elif sm == 'laplacian':
                mesh = mesh.filter_smooth_laplacian(number_of_iterations=int(smooth_iterations))
                print(f"  平滑: laplacian(iter={smooth_iterations})")
            elif sm == 'taubin':
                mesh = mesh.filter_smooth_taubin(number_of_iterations=int(smooth_iterations))
                print(f"  平滑: taubin(iter={smooth_iterations})")
            else:
                pass
            # 平滑后需要重新计算法向
            try:
                mesh.compute_vertex_normals()
            except Exception:
                pass
        except AttributeError:
            print("  平滑: Open3D 版本不支持所选平滑API，跳过")
        except Exception as e:
            warnings.warn(f"  平滑: 失败，原因: {e}")
        
        # 保存为STL文件
        success = o3d.io.write_triangle_mesh(output_path, mesh)
        
        if success:
            print(f"成功重建: {pointcloud_file_path} -> {output_path}")
            print(f"点云点数: {len(points)}")
            print(f"重建网格 - 顶点数: {len(mesh.vertices)}, 面数: {len(mesh.triangles)}")
            print("-" * 50)
            return True
        else:
            print(f"保存STL文件失败: {output_path}")
            return False
            
    except Exception as e:
        print(f"处理文件 {pointcloud_file_path} 时出错: {str(e)}")
        return False


def reconstruct_poisson(pcd, depth=10, width=0, scale=1.1, linear_fit=False,
                        density_quantile=0.01, fill_holes=True):
    """
    使用泊松重建方法重建网格
    
    Args:
        pcd: Open3D点云对象
        depth: 八叉树深度
        width: 宽度参数（Open3D 接口保留）
        scale: 尺度参数
        linear_fit: 是否使用线性拟合
        density_quantile: 密度分位数阈值
        fill_holes: 是否填充孔洞
    
    Returns:
        Open3D网格对象
    """
    # 确保深度为有效值；默认使用10
    if depth is None or depth <= 0:
        depth = 10

    print(f"使用泊松重建 (深度={depth}, 尺度={scale})")

    # 执行泊松重建（CPU）
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit
    )
    
    # 计算法向量（保存STL需要）
    mesh.compute_vertex_normals()
    
    # 根据密度移除噪声顶点（密度阈值用于去除重建的杂乱部分）
    if density_quantile is not None and 0 < float(density_quantile) < 1:
        try:
            threshold = np.quantile(densities, float(density_quantile))
            mesh.remove_vertices_by_mask(densities < threshold)
            print(f"  低密度剔除: 分位数={float(density_quantile)}")
        except Exception as e:
            warnings.warn(f"  低密度剔除失败: {e}")

    # 网格清理
    try:
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        mesh.remove_non_manifold_edges()
    except Exception:
        pass

    if fill_holes:
        try:
            mesh = fill_mesh_holes_tensor(mesh)
            print("  首次补洞: 成功使用 tensor.fill_holes(hole_size=..) 并回写 legacy")
        except Exception as e:
            warnings.warn(f"  首次补洞: 失败，原因: {e}")

    return mesh


def reconstruct_ball_pivoting(pcd, radius):
    """
    使用球旋转重建方法重建网格
    
    Args:
        pcd: Open3D点云对象
        radius: 球旋转半径
    
    Returns:
        Open3D网格对象
    """
    print(f"使用球旋转重建 (半径={radius})")
    
    # 如果未指定半径，自动估算
    if radius <= 0:
        # 计算点云的平均距离作为半径
        distances = pcd.compute_nearest_neighbor_distance()
        radius = np.mean(distances) * 2
    
    # 执行球旋转重建
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
            print(f"  自动估算 alpha: {alpha:.6f}")
        except Exception:
            alpha = 0.01
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    try:
        mesh.compute_vertex_normals()
    except Exception:
        pass
    # 清理
    try:
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
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


def process_directory(
    input_dir,
    output_dir,
    method='poisson',
    depth=10,
    width=0,
    scale=1.1,
    linear_fit=False,
    density_quantile=0.01,
    fill_holes=True,
    remove_outliers=True,
    nb_neighbors=20,
    std_ratio=2.0,
    every_k_points=None,
    voxel_size=None,
    recompute_normals=True,
    orient_k=50,
    cluster_cleanup=False,
    cluster_eps_scale=1.5,
    cluster_min_points=50,
    planarize=False,
    planarize_strength=1.0,
    align_to_z=True,
    smooth_method='taubin',
    smooth_iterations=50,
    decimate_target_triangles=0,
):
    """
    处理目录中的所有点云文件
    
    Args:
        input_dir (str): 输入目录路径
        output_dir (str): 输出目录路径
        method (str): 重建方法
        depth (int): 八叉树深度
        width (float): 球旋转半径 / AlphaShape alpha
        scale (float): 泊松重建尺度参数
        linear_fit (bool): 是否使用线性拟合
        density_quantile (float): 泊松重建后低密度剔除分位数
        fill_holes (bool): 是否补洞
        remove_outliers (bool): 是否移除统计离群点
        every_k_points, voxel_size: 下采样
        recompute_normals, orient_k: 法向估计与一致性
        smooth_method, smooth_iterations: 网格平滑
        decimate_target_triangles: 网格简化目标三角形数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有.npy点云文件
    pointcloud_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith('.npy'):
            pointcloud_files.append(os.path.join(input_dir, file))
    
    if not pointcloud_files:
        print(f"在目录 {input_dir} 中未找到.npy文件")
        return
    
    print(f"找到 {len(pointcloud_files)} 个点云文件")
    
    # 处理每个文件
    success_count = 0
    for pointcloud_file in tqdm(pointcloud_files, desc="处理点云文件"):
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(pointcloud_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}.stl")
        
        # 重建STL
        if pointcloud_to_stl(
            pointcloud_file,
            output_file,
            method,
            depth,
            width,
            scale,
            linear_fit,
            density_quantile,
            fill_holes,
            remove_outliers,
            nb_neighbors,
            std_ratio,
            every_k_points,
            voxel_size,
            recompute_normals,
            orient_k,
            cluster_cleanup,
            cluster_eps_scale,
            cluster_min_points,
            planarize,
            planarize_strength,
            align_to_z,
            smooth_method,
            smooth_iterations,
            decimate_target_triangles,
        ):
            success_count += 1
    
    print(f"\n处理完成: {success_count}/{len(pointcloud_files)} 个文件成功重建")


def main():
    parser = argparse.ArgumentParser(description='从点云重建STL文件')
    parser.add_argument('input', nargs='?', default='output/pointcloud/insoles',
                       help='输入点云文件路径或目录 (默认: output/pointcloud/insoles)')
    parser.add_argument('output', nargs='?', default='output/raw/insoles',
                       help='输出STL文件路径或目录 (默认: output/raw/insoles)')
    parser.add_argument('--method', choices=['poisson', 'ball_pivoting', 'alpha_shape'], 
                       default='poisson', help='重建方法 (默认: poisson)')
    parser.add_argument('--depth', type=int, default=10, 
                       help='泊松重建的八叉树深度 (建议: 8-10; 默认: 10)')
    parser.add_argument('--width', type=float, default=0, 
                       help='球旋转半径 / AlphaShape alpha (默认: 自动估算/0)')
    parser.add_argument('--scale', type=float, default=1.1, 
                       help='泊松重建尺度参数 (默认: 1.1)')
    parser.add_argument('--linear-fit', action='store_true', 
                       help='使用线性拟合 (泊松重建)')

    # 预处理与法向
    parser.add_argument('--every-k', type=int, default=0, help='均匀下采样的 every_k_points，<=1 禁用')
    parser.add_argument('--voxel-size', type=float, default=0.0, help='体素下采样大小，<=0 禁用')
    parser.add_argument('--no-outlier-removal', action='store_true', help='禁用统计离群点移除')
    parser.add_argument('--outlier-nb-neighbors', type=int, default=20, help='统计离群点移除的邻居数 (默认: 20)')
    parser.add_argument('--outlier-std-ratio', type=float, default=2.0, help='统计离群点移除的标准差阈值 (默认: 2.0)')
    parser.add_argument('--cluster-cleanup', action='store_true', help='启用基于聚类的游离点清理（默认关闭）')
    parser.add_argument('--cluster-eps-scale', type=float, default=1.5, help='DBSCAN eps = mean_nn * scale (默认: 1.5)')
    parser.add_argument('--cluster-min-points', type=int, default=50, help='DBSCAN min_points (默认: 50)')
    parser.add_argument('--planarize', action='store_true', help='启用平面化处理（默认关闭）')
    parser.add_argument('--planarize-strength', type=float, default=1.0, help='平面化强度[0,1]，1表示完全投影到平面')
    parser.add_argument('--no-align-to-z', action='store_true', help='平面化时不强制对齐Z轴')
    parser.add_argument('--no-recompute-normals', action='store_true', help='禁用预处理后的法向重估计与一致性朝向')
    parser.add_argument('--orient-k', type=int, default=50, help='法向一致性邻域K (默认: 50)')

    # 泊松后处理与网格后处理
    parser.add_argument('--density-quantile', type=float, default=0.01, help='密度分位数阈值，移除低密度区域 (默认: 0.01)')
    parser.add_argument('--no-fill-holes', action='store_true', help='禁用填洞')
    parser.add_argument('--smooth-method', choices=['none', 'simple', 'laplacian', 'taubin'], default='taubin', help='网格平滑方法')
    parser.add_argument('--smooth-iterations', type=int, default=50, help='平滑迭代次数 (默认: 50)')
    parser.add_argument('--decimate-target-triangles', type=int, default=0, help='网格简化目标三角形数（<=0 禁用）')

    args = parser.parse_args()
    
    # 检查输入路径
    if not os.path.exists(args.input):
        print(f"错误: 输入路径不存在: {args.input}")
        return
    
    # 处理单个文件或目录
    if os.path.isfile(args.input):
        if not args.input.lower().endswith('.npy'):
            print("错误: 输入文件必须是.npy格式")
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
            args.density_quantile,
            not args.no_fill_holes,
            not args.no_outlier_removal,
            args.outlier_nb_neighbors,
            args.outlier_std_ratio,
            args.every_k if args.every_k > 1 else None,
            args.voxel_size if args.voxel_size and args.voxel_size > 0 else None,
            not args.no_recompute_normals,
            args.orient_k,
            args.cluster_cleanup,
            args.cluster_eps_scale,
            args.cluster_min_points,
            args.planarize,
            args.planarize_strength,
            not args.no_align_to_z,
            args.smooth_method,
            args.smooth_iterations,
            args.decimate_target_triangles,
        )
    
    elif os.path.isdir(args.input):
        process_directory(
            args.input,
            args.output,
            method=args.method,
            depth=args.depth,
            width=args.width,
            scale=args.scale,
            linear_fit=args.linear_fit,
            density_quantile=args.density_quantile,
            fill_holes=not args.no_fill_holes,
            remove_outliers=not args.no_outlier_removal,
            nb_neighbors=args.outlier_nb_neighbors,
            std_ratio=args.outlier_std_ratio,
            every_k_points=(args.every_k if args.every_k > 1 else None),
            voxel_size=(args.voxel_size if args.voxel_size and args.voxel_size > 0 else None),
            recompute_normals=not args.no_recompute_normals,
            orient_k=args.orient_k,
            cluster_cleanup=args.cluster_cleanup,
            cluster_eps_scale=args.cluster_eps_scale,
            cluster_min_points=args.cluster_min_points,
            planarize=args.planarize,
            planarize_strength=args.planarize_strength,
            align_to_z=not args.no_align_to_z,
            smooth_method=args.smooth_method,
            smooth_iterations=args.smooth_iterations,
            decimate_target_triangles=args.decimate_target_triangles,
        )
    
    else:
        print(f"错误: 无效的输入路径: {args.input}")


if __name__ == "__main__":
    main()

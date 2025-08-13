from typing import Any
import numpy as np
import os
import glob
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings
import io
from contextlib import redirect_stderr, redirect_stdout, nullcontext


def _import_open3d():
    try:
        import open3d as o3d  # type: ignore
        return o3d
    except Exception as e:
        raise RuntimeError(
            "Open3D 未安装或与当前 Python 版本不兼容。建议使用 Python 3.10-3.12 并安装 open3d>=0.17。"
        ) from e


def pointcloud_to_stl(
    pointcloud_file_path,
    output_path,
    method='poisson',
    depth=12,
    width=0,
    scale=1.1,
    linear_fit=False,
    density_quantile=0.0,
    fill_holes=True,
    remove_outliers=False,
    nb_neighbors=30,
    std_ratio=3.0,
    radius_removal=False,
    radius=0.0,
    radius_nb_points=16,
    cluster_cleanup=False,
    cluster_eps_scale=1.5,
    cluster_min_points=50,
    planarize=False,
    planarize_strength=1.0,
    align_to_z=True,
    orient_normals=True,
    orient_k=50,
    voxel_size=0.0,
    smooth_mesh=True,
    smooth_method='taubin',
    smooth_iterations=12,
    smooth_lambda=0.5,
    keep_largest_component=False,
    lcc_min_triangles=200,
    orient_triangles=True,
    suppress_native_warnings=False,
    clip_to_input_bounds=True,
    clip_margin_scale=2.0,
    remove_small_components=True,
    min_component_area_ratio=0.02,
    trim_by_distance=True,
    trim_distance_multiplier=3.0,
):
    """
    从点云文件重建STL网格并保存
    
    Args:
        pointcloud_file_path (str): 点云文件路径（.npy格式）
        output_path (str): 输出STL文件路径
        method (str): 重建方法，'poisson'、'ball_pivoting'、'alpha_shape' 或 'auto'
        depth (int): 八叉树深度，用于泊松重建；<=0 时自动估算
        width (float): 球旋转半径/Alpha值（按方法解释）
        scale (float): 泊松重建的尺度参数
        linear_fit (bool): 是否使用线性拟合
        density_quantile (float): 密度分位数阈值，用于移除噪声
        fill_holes (bool): 是否填充网格孔洞
    """
    try:
        o3d = _import_open3d()
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
        
        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 可选：体素下采样，统一密度，增强鲁棒性
        if voxel_size is not None and float(voxel_size) > 0:
            try:
                pcd = pcd.voxel_down_sample(float(voxel_size))
                print(f"  已体素下采样 (voxel_size={float(voxel_size):.6f}) | 点数: {len(pcd.points)}")
                normals = None  # 下采样后作废外部法向
            except Exception as e_vx:
                warnings.warn(f"体素下采样失败: {e_vx}")
        
        # 法向处理：估计 + 一致性定向
        try:
            if normals is not None and (voxel_size is None or float(voxel_size) <= 0):
                pcd.normals = o3d.utility.Vector3dVector(normals)
            else:
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
            if orient_normals:
                # 使用一致切平面方法做法向全局一致
                k = max(10, int(orient_k))
                pcd.orient_normals_consistent_tangent_plane(k)
            pcd.normalize_normals()
        except Exception as e_n:
            warnings.warn(f"法向估计/定向失败，将继续: {e_n}")
        
        # 预处理：移除游离点 / 异常点（统计）
        if remove_outliers:
            try:
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
                print(f"  已移除统计离群点 (nb_neighbors={nb_neighbors}, std_ratio={std_ratio})")
            except Exception as e_rm:
                warnings.warn(f"统计离群点移除失败: {e_rm}")

        # 预处理：半径离群点移除（可选）
        if radius_removal:
            try:
                if radius is None or float(radius) <= 0:
                    nn_dists = pcd.compute_nearest_neighbor_distance()
                    mean_nn = float(np.mean(nn_dists)) if len(nn_dists) > 0 else 0.0
                    rad = mean_nn * 2.0 if mean_nn > 0 else 0.01
                else:
                    rad = float(radius)
                min_nb = max(4, int(radius_nb_points))
                pcd, _ = pcd.remove_radius_outlier(nb_points=min_nb, radius=rad)
                print(f"  已移除半径离群点 (radius={rad:.6f}, nb_points>={min_nb}) | 点数: {len(pcd.points)}")
            except Exception as e_rr:
                warnings.warn(f"半径离群点移除失败: {e_rr}")

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
                # 重新估算并定向法向
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
                if orient_normals:
                    pcd.orient_normals_consistent_tangent_plane(max(10, int(orient_k)))
                pcd.normalize_normals()
                print("  已执行平面化处理并对齐Z轴")
            except Exception as e_pl:
                warnings.warn(f"平面化处理失败: {e_pl}")

        # 选择并执行网格重建
        method_lower = str(method).lower()
        if method_lower == 'auto':
            # 简单平面度检测：最小特征值比例越小越接近平面
            try:
                pts_np = np.asarray(pcd.points)
                centroid = pts_np.mean(axis=0)
                centered = pts_np - centroid
                cov = centered.T @ centered / max(1, centered.shape[0] - 1)
                evals, _ = np.linalg.eigh(cov)
                evals = np.clip(evals, 1e-12, None)
                planarity = float(evals.min() / evals.sum()) if evals.sum() > 0 else 0.0
            except Exception:
                planarity = 0.0
            chosen = 'alpha_shape' if planarity < 0.02 else 'poisson'
            print(f"  自动选择重建方法: {chosen} (planarity={planarity:.6f})")
            method_lower = chosen

        if method_lower == 'poisson':
            mesh = reconstruct_poisson(
                pcd,
                depth,
                width,
                scale,
                linear_fit,
                density_quantile=density_quantile,
                fill_holes=fill_holes,
                suppress_native_warnings=suppress_native_warnings,
            )
        elif method_lower == 'ball_pivoting':
            mesh = reconstruct_ball_pivoting(pcd, width)
        elif method_lower == 'alpha_shape':
            # 将 width 解释为 alpha；若未提供则基于平均邻域估计
            mesh = reconstruct_alpha_shape(pcd, alpha=width)
        else:
            raise ValueError(f"不支持的重建方法: {method}")

        # 计算输入点云边界与裁剪margin
        pts_np_for_bounds = np.asarray(pcd.points)
        bounds_min = pts_np_for_bounds.min(axis=0)
        bounds_max = pts_np_for_bounds.max(axis=0)
        # 估计邻域尺度
        try:
            nn_dists = pcd.compute_nearest_neighbor_distance()
            mean_nn = float(np.mean(nn_dists)) if len(nn_dists) > 0 else 0.0
        except Exception:
            mean_nn = 0.0
        diag = float(np.linalg.norm(bounds_max - bounds_min))
        default_margin = diag * 0.01 if diag > 0 else 1e-3
        clip_margin = float(clip_margin_scale) * mean_nn if mean_nn > 0 else default_margin

        # 网格后处理：裁剪、清理、连通片、平滑、洞填补、方向
        mesh = postprocess_mesh(
            mesh,
            fill_holes=fill_holes,
            smooth_mesh=smooth_mesh,
            smooth_method=smooth_method,
            smooth_iterations=smooth_iterations,
            smooth_lambda=smooth_lambda,
            keep_largest_component=keep_largest_component,
            lcc_min_triangles=lcc_min_triangles,
            orient_triangles=orient_triangles,
            clip_to_input_bounds=clip_to_input_bounds,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            clip_margin=clip_margin,
            remove_small_components=remove_small_components,
            min_component_area_ratio=min_component_area_ratio,
            reference_points=pts_np_for_bounds,
            mean_nn_distance=mean_nn,
            trim_by_distance=trim_by_distance,
            trim_distance_multiplier=trim_distance_multiplier,
        )
        
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
                        density_quantile=0.01, fill_holes=True, suppress_native_warnings: bool = False):
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

    print(f"使用泊松重建 (深度={depth}, 尺度={scale})")

    # 执行泊松重建
    # 注意：Open3D的泊松重建在此仅使用CPU
    o3d = _import_open3d()
    # 可选：捕获底层 PoissonRecon 的控制台输出（bad average roots 等），减少噪声
    capture_stream = io.StringIO()
    ctx = (redirect_stderr(capture_stream), redirect_stdout(capture_stream)) if suppress_native_warnings else (nullcontext(), nullcontext())
    # 同时降低 Open3D 的日志级别到 Error（若可用）
    prev_level = None
    if suppress_native_warnings:
        try:
            prev_level = o3d.utility.get_verbosity_level()
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        except Exception:
            prev_level = None
    try:
        with ctx[0]:
            with ctx[1]:
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit
                )
    finally:
        if suppress_native_warnings and prev_level is not None:
            try:
                o3d.utility.set_verbosity_level(prev_level)
            except Exception:
                pass
    
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
        # Open3D >= 0.17 provides fill_holes; 若不可用则忽略
        try:
            mesh = mesh.fill_holes()
        except AttributeError:
            pass

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
    print(f"使用球旋转重建 (半径={radius})")
    
    # 如果未指定半径，自动估算
    if radius <= 0:
        # 计算点云的平均距离作为半径
        distances = pcd.compute_nearest_neighbor_distance()
        radius = np.mean(distances) * 2
    
    # 执行球旋转重建
    # 注意：球旋转重建目前主要在CPU上实现
    o3d = _import_open3d()
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
    o3d = _import_open3d()
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


def postprocess_mesh(
    mesh: Any,
    fill_holes: bool = True,
    smooth_mesh: bool = True,
    smooth_method: str = 'taubin',
    smooth_iterations: int = 10,
    smooth_lambda: float = 0.5,
    keep_largest_component: bool = True,
    lcc_min_triangles: int = 200,
    orient_triangles: bool = True,
    clip_to_input_bounds: bool = True,
    bounds_min: Any | None = None,
    bounds_max: Any | None = None,
    clip_margin: float = 0.0,
    remove_small_components: bool = True,
    min_component_area_ratio: float = 0.02,
    reference_points: Any | None = None,
    mean_nn_distance: float | None = None,
    trim_by_distance: bool = True,
    trim_distance_multiplier: float = 3.0,
) -> Any:
    """通用网格后处理，提升完整性与光滑度。"""
    o3d = _import_open3d()
    try:
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        mesh.remove_non_manifold_edges()
    except Exception:
        pass

    # 先按输入点云边界裁剪，去掉超出AABB+margin的冗余表面（常见于Poisson包络环）
    if clip_to_input_bounds and bounds_min is not None and bounds_max is not None:
        try:
            v = np.asarray(mesh.vertices)
            t = np.asarray(mesh.triangles)
            lo = np.asarray(bounds_min) - float(clip_margin)
            hi = np.asarray(bounds_max) + float(clip_margin)
            tri_vertices = v[t]
            # 三角形任一顶点在边界外则移除
            outside = np.logical_or(tri_vertices < lo, tri_vertices > hi).any(axis=(1, 2))
            if outside.any():
                mesh.remove_triangles_by_mask(outside)
                mesh.remove_unreferenced_vertices()
                print(f"  已按输入AABB裁剪三角形: {int(outside.sum())}")
        except Exception as e_clip:
            warnings.warn(f"AABB裁剪失败: {e_clip}")

    if keep_largest_component:
        try:
            triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
            if len(cluster_n_triangles) > 0:
                largest_idx = int(np.argmax(cluster_n_triangles))
                labels = np.asarray(triangle_clusters)
                # 保留“最大连通片 + 所有三角形数≥阈值的连通片”
                if lcc_min_triangles is not None and int(lcc_min_triangles) > 0:
                    keep_set = {largest_idx}
                    keep_set.update([idx for idx, ntri in enumerate(cluster_n_triangles) if int(ntri) >= int(lcc_min_triangles)])
                    remove_mask = np.isin(labels, list(keep_set), invert=True)
                    if remove_mask.sum() > 0:
                        mesh.remove_triangles_by_mask(remove_mask)
                        mesh.remove_unreferenced_vertices()
                    print(f"  已保留 {len(keep_set)} 个连通片 (>= {int(lcc_min_triangles)} triangles; 最大片={int(cluster_n_triangles[largest_idx])})")
                else:
                    mask = labels != largest_idx
                    if mask.sum() > 0:
                        mesh.remove_triangles_by_mask(mask)
                        mesh.remove_unreferenced_vertices()
                    print(f"  已保留最大连通片 (triangles={int(cluster_n_triangles[largest_idx])})")
        except Exception as e_cc:
            warnings.warn(f"连通片聚类失败: {e_cc}")

    # 按顶点到输入点云的最近距离剔除远离的三角形（常见包络环通常远离原始点云）
    if trim_by_distance and reference_points is not None:
        try:
            o3d = _import_open3d()
            ref_pcd = o3d.geometry.PointCloud()
            ref_pcd.points = o3d.utility.Vector3dVector(np.asarray(reference_points))
            kdtree = o3d.geometry.KDTreeFlann(ref_pcd)
            v = np.asarray(mesh.vertices)
            n_vertices = v.shape[0]
            dists = np.empty(n_vertices, dtype=np.float64)
            # 逐点查询最近邻距离
            for i in range(n_vertices):
                _, idx, sq = kdtree.search_knn_vector_3d(v[i], 1)
                if len(sq) > 0:
                    dists[i] = float(np.sqrt(sq[0]))
                else:
                    dists[i] = np.inf
            # 阈值：mean_nn * multiplier（兜底用对角线的1%）
            if mean_nn_distance is None or mean_nn_distance <= 0:
                ref = np.asarray(reference_points)
                diag = float(np.linalg.norm(ref.max(axis=0) - ref.min(axis=0)))
                base = diag * 0.01
            else:
                base = float(mean_nn_distance)
            thr = base * float(trim_distance_multiplier)
            t = np.asarray(mesh.triangles)
            tri_thr = np.all(dists[t] > thr, axis=1)
            if tri_thr.any():
                mesh.remove_triangles_by_mask(tri_thr)
                mesh.remove_unreferenced_vertices()
                print(f"  已按距离阈值剔除三角形: {int(tri_thr.sum())} (thr={thr:.6f})")
        except Exception as e_trim:
            warnings.warn(f"距离裁剪失败: {e_trim}")

    # 按连通片面积比例剔除很小片段（在不启用 keep_largest_component 的情况下也可工作）
    if remove_small_components:
        try:
            labels, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
            labels = np.asarray(labels)
            if labels.size > 0:
                v = np.asarray(mesh.vertices)
                t = np.asarray(mesh.triangles)
                tri = v[t]
                tri_areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
                # 聚合每个簇的面积
                num_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
                if num_clusters > 0:
                    cluster_areas = np.zeros(num_clusters, dtype=np.float64)
                    for idx in range(num_clusters):
                        cluster_areas[idx] = float(tri_areas[labels == idx].sum())
                    if cluster_areas.size > 0 and cluster_areas.max() > 0:
                        largest_area = float(cluster_areas.max())
                        # 需要移除的簇：面积比例过小 且 三角形数也较少
                        remove_clusters = [idx for idx, area in enumerate(cluster_areas)
                                           if (area / largest_area) < float(min_component_area_ratio)
                                           and int(cluster_n_triangles[idx]) < int(lcc_min_triangles)]
                        if len(remove_clusters) > 0:
                            remove_mask = np.isin(labels, np.array(remove_clusters, dtype=np.int32))
                            mesh.remove_triangles_by_mask(remove_mask)
                            mesh.remove_unreferenced_vertices()
                            print(f"  已按面积比例剔除连通片: {len(remove_clusters)} (ratio<{float(min_component_area_ratio):.3f})")
        except Exception as e_small:
            warnings.warn(f"小连通片剔除失败: {e_small}")

    if fill_holes:
        try:
            mesh = mesh.fill_holes()
            print("  已尝试填充孔洞")
        except Exception:
            pass

    if smooth_mesh:
        try:
            if smooth_method == 'taubin':
                mesh = mesh.filter_smooth_taubin(number_of_iterations=int(smooth_iterations))
            elif smooth_method == 'laplacian':
                # 某些Open3D版本允许 lambda 参数；如不可用则忽略
                try:
                    mesh = mesh.filter_smooth_laplacian(number_of_iterations=int(smooth_iterations), lambda_factor=float(smooth_lambda))
                except TypeError:
                    mesh = mesh.filter_smooth_laplacian(number_of_iterations=int(smooth_iterations))
            else:
                mesh = mesh.filter_smooth_simple(number_of_iterations=int(smooth_iterations))
            mesh.compute_vertex_normals()
            print(f"  已平滑网格 ({smooth_method}, iters={smooth_iterations})")
        except Exception as e_sm:
            warnings.warn(f"平滑失败: {e_sm}")

    if orient_triangles:
        try:
            mesh.orient_triangles()
        except Exception:
            pass

    # 最终清理一次
    try:
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    return mesh


def process_directory(input_dir, output_dir, method='poisson', 
                     depth=10, width=0, scale=1.1, linear_fit=False, device='CPU:0', **kwargs):
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
        **kwargs: 透传至 pointcloud_to_stl 的其他参数
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
        if pointcloud_to_stl(pointcloud_file, output_file, method, depth, width, scale, linear_fit, **kwargs):
            success_count += 1
    
    print(f"\n处理完成: {success_count}/{len(pointcloud_files)} 个文件成功重建")


def main():
    parser = argparse.ArgumentParser(description='从点云重建STL文件')
    parser.add_argument('input', nargs='?', default='output/pointcloud/insoles',
                       help='输入点云文件路径或目录 (默认: output/pointcloud/insoles)')
    parser.add_argument('output', nargs='?', default='output/raw/insoles',
                       help='输出STL文件路径或目录 (默认: output/raw/insoles)')
    parser.add_argument('--method', choices=['auto', 'poisson', 'ball_pivoting', 'alpha_shape'], 
                       default='poisson', help='重建方法 (默认: poisson)')
    parser.add_argument('--depth', type=int, default=12, 
                       help='泊松重建的八叉树深度 (默认: 12)')
    parser.add_argument('--width', type=float, default=0, 
                         help='球旋转半径 / Alpha 值 (默认: 自动估算)')
    parser.add_argument('--scale', type=float, default=1.1, 
                        help='泊松重建尺度参数 (默认: 1.1)')
    parser.add_argument('--linear-fit', action='store_true', 
                       help='使用线性拟合 (泊松重建)')

    # 预处理与法向
    parser.add_argument('--no-outlier-removal', action='store_true', help='禁用统计离群点移除')
    parser.add_argument('--outlier-nb-neighbors', type=int, default=30, help='统计离群点移除的邻居数')
    parser.add_argument('--outlier-std-ratio', type=float, default=3.0, help='统计离群点移除的标准差阈值')
    parser.add_argument('--radius-removal', action='store_true', help='启用半径离群点移除（默认关闭）')
    parser.add_argument('--radius', type=float, default=0.0, help='半径离群点移除的半径；<=0 自动估计')
    parser.add_argument('--radius-nb-points', type=int, default=16, help='半径离群点移除的最小邻居数')
    parser.add_argument('--voxel-size', type=float, default=0.0, help='体素下采样大小（0 表示禁用）')
    parser.add_argument('--orient-normals', dest='orient_normals', action='store_true', help='启用法向一致性定向')
    parser.add_argument('--no-orient-normals', dest='orient_normals', action='store_false', help='禁用法向一致性定向')
    parser.set_defaults(orient_normals=True)
    parser.add_argument('--orient-k', type=int, default=50, help='法向一致性定向的K邻居数')

    # 聚类与平面
    parser.add_argument('--cluster-cleanup', action='store_true', help='启用基于聚类的游离点清理（默认关闭）')
    parser.add_argument('--cluster-eps-scale', type=float, default=1.5, help='DBSCAN eps 缩放系数（均值邻域 * scale）')
    parser.add_argument('--cluster-min-points', type=int, default=50, help='DBSCAN 最小点数')
    parser.add_argument('--planarize', action='store_true', help='启用平面化处理（默认关闭）')
    parser.add_argument('--planarize-strength', type=float, default=1.0, help='平面化强度[0,1]，1表示完全投影到平面')
    parser.add_argument('--align-to-z', dest='align_to_z', action='store_true', help='将主法向对齐到 +Z 方向')
    parser.add_argument('--no-align-to-z', dest='align_to_z', action='store_false', help='不对齐到 +Z')
    parser.set_defaults(align_to_z=True)

    # 泊松密度过滤与洞填补
    parser.add_argument('--density-quantile', type=float, default=0.0, help='按密度分位数移除低密度区域（默认关闭）')
    parser.add_argument('--fill-holes', dest='fill_holes', action='store_true', help='填充孔洞（默认启用）')
    parser.add_argument('--no-fill-holes', dest='fill_holes', action='store_false', help='禁用孔洞填充')
    parser.set_defaults(fill_holes=True)

    # 网格后处理
    parser.add_argument('--smooth-mesh', dest='smooth_mesh', action='store_true', help='启用网格平滑（默认启用）')
    parser.add_argument('--no-smooth-mesh', dest='smooth_mesh', action='store_false', help='禁用网格平滑')
    parser.set_defaults(smooth_mesh=True)
    parser.add_argument('--smooth-method', choices=['taubin', 'laplacian', 'simple'], default='taubin', help='平滑方法')
    parser.add_argument('--smooth-iterations', type=int, default=12, help='平滑迭代次数（默认 12）')
    parser.add_argument('--smooth-lambda', type=float, default=0.5, help='Laplacian 平滑的lambda因子（部分版本可能忽略）')
    parser.add_argument('--keep-largest-component', dest='keep_largest_component', action='store_true', help='仅保留最大连通片（默认关闭）')
    parser.add_argument('--no-keep-largest-component', dest='keep_largest_component', action='store_false', help='禁用仅保留最大连通片')
    parser.set_defaults(keep_largest_component=False)
    parser.add_argument('--lcc-min-triangles', type=int, default=200, help='保留最大连通片时的最小三角形数阈值（参考）')
    parser.add_argument('--orient-triangles', dest='orient_triangles', action='store_true', help='统一三角形朝向')
    parser.add_argument('--no-orient-triangles', dest='orient_triangles', action='store_false', help='不统一三角形朝向')
    parser.set_defaults(orient_triangles=True)
    parser.add_argument('--suppress-native-warnings', dest='suppress_native_warnings', action='store_true', help='尽量捕获底层 PoissonRecon 的控制台警告')
    parser.add_argument('--no-suppress-native-warnings', dest='suppress_native_warnings', action='store_false', help='不捕获底层 PoissonRecon 控制台输出')
    parser.set_defaults(suppress_native_warnings=False)
    # 剪裁与小片段移除
    parser.add_argument('--clip-to-input-bounds', dest='clip_to_input_bounds', action='store_true', help='裁剪超出输入点云AABB+margin的三角形（默认启用）')
    parser.add_argument('--no-clip-to-input-bounds', dest='clip_to_input_bounds', action='store_false', help='不进行AABB裁剪')
    parser.set_defaults(clip_to_input_bounds=True)
    parser.add_argument('--clip-margin-scale', type=float, default=2.0, help='AABB裁剪的邻域半径倍数（缺省用对角线1%%）')
    parser.add_argument('--remove-small-components', dest='remove_small_components', action='store_true', help='按面积比例剔除很小连通片（默认启用）')
    parser.add_argument('--no-remove-small-components', dest='remove_small_components', action='store_false', help='不剔除小连通片')
    parser.set_defaults(remove_small_components=True)
    parser.add_argument('--min-component-area-ratio', type=float, default=0.02, help='相对最大面的最小连通片面积比例阈值')
    parser.add_argument('--trim-by-distance', dest='trim_by_distance', action='store_true', help='按顶点到输入点云的最近距离剔除远离的三角形（默认启用）')
    parser.add_argument('--no-trim-by-distance', dest='trim_by_distance', action='store_false', help='不进行距离裁剪')
    parser.set_defaults(trim_by_distance=True)
    parser.add_argument('--trim-distance-multiplier', type=float, default=3.0, help='距离裁剪阈值=均值邻域距离×该倍率（兜底用对角线1%%）')

    # 兼容下方对 args.device 的引用（目前重建过程仅使用CPU，此参数仅为接口一致性）
    parser.add_argument('--device', type=str, default='cpu', help='计算设备（当前未使用，仅为兼容接口）')
    
    args = parser.parse_args()
    
    # 检查输入路径
    if not os.path.exists(args.input):
        print(f"错误: 输入路径不存在: {args.input}")
        return
    
    # 收集通用参数
    common_kwargs = dict(
        method=args.method,
        depth=args.depth,
        width=args.width,
        scale=args.scale,
        linear_fit=args.linear_fit,
        density_quantile=args.density_quantile,
        fill_holes=args.fill_holes,
        remove_outliers=not args.no_outlier_removal,
        nb_neighbors=args.outlier_nb_neighbors,
        std_ratio=args.outlier_std_ratio,
        radius_removal=args.radius_removal,
        radius=args.radius,
        radius_nb_points=args.radius_nb_points,
        cluster_cleanup=args.cluster_cleanup,
        cluster_eps_scale=args.cluster_eps_scale,
        cluster_min_points=args.cluster_min_points,
        planarize=args.planarize,
        planarize_strength=args.planarize_strength,
        align_to_z=args.align_to_z,
        orient_normals=args.orient_normals,
        orient_k=args.orient_k,
        voxel_size=args.voxel_size,
        smooth_mesh=args.smooth_mesh,
        smooth_method=args.smooth_method,
        smooth_iterations=args.smooth_iterations,
        smooth_lambda=args.smooth_lambda,
        keep_largest_component=args.keep_largest_component,
        lcc_min_triangles=args.lcc_min_triangles,
        orient_triangles=args.orient_triangles,
        suppress_native_warnings=args.suppress_native_warnings,
        clip_to_input_bounds=args.clip_to_input_bounds,
        clip_margin_scale=args.clip_margin_scale,
        remove_small_components=args.remove_small_components,
        min_component_area_ratio=args.min_component_area_ratio,
        trim_by_distance=args.trim_by_distance,
        trim_distance_multiplier=args.trim_distance_multiplier,
    )
    
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
            **common_kwargs,
        )
    
    elif os.path.isdir(args.input):
        process_directory(
            args.input,
            args.output,
            device=args.device,
            **common_kwargs,
        )
    
    else:
        print(f"错误: 无效的输入路径: {args.input}")


if __name__ == "__main__":
    main()

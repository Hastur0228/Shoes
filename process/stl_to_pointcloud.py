import open3d as o3d
import numpy as np
import os
import glob
from pathlib import Path
import argparse

# PCA 对齐与归一化工具（兼容作为包运行或直接脚本运行）
try:  # python -m process.stl_to_pointcloud
    from .pca_align import PCAAlignConfig, pca_align_and_normalize
except Exception:  # python process/stl_to_pointcloud.py
    from pca_align import PCAAlignConfig, pca_align_and_normalize

 


def stl_to_pointcloud(
    stl_file_path,
    output_path,
    num_points=300000,
    uniform_sampling=True,
    clean_mesh: bool = True,
    # 预处理：PCA 对齐、原点统一与尺度归一化
    enable_pca_align: bool = True,
    pca_origin: str = "heel",  # 'heel' | 'bottom'
    pca_toe_positive_x: bool = True,
    pca_heel_slice_ratio: float = 0.08,
    pca_scale_mode: str = "none",  # 'none' | 'length' | 'width'
    pca_target_length: float = 1.0,
    pca_target_width: float = 1.0,
):
    """
    从STL文件中采样点云并保存为.npy格式
    
    Args:
        stl_file_path (str): STL文件路径
        output_path (str): 输出.npy文件路径
        num_points (int): 采样点数量，默认300,000
        uniform_sampling (bool): 是否使用均匀采样，默认True(使用均匀采样)
    """
    try:
        # 读取STL文件
        mesh = o3d.io.read_triangle_mesh(stl_file_path)
        
        # 确保网格是有效的
        if not mesh.has_vertices():
            print(f"警告: {stl_file_path} 没有顶点数据")
            return False
        
        # 获取原始顶点数和面数
        original_vertices = len(mesh.vertices)
        original_triangles = len(mesh.triangles)
            
        # 清理网格（有助于避免采样阶段的 GPU 内核断言）
        if clean_mesh:
            try:
                mesh.remove_duplicated_vertices()
                mesh.remove_duplicated_triangles()
                mesh.remove_degenerate_triangles()
                mesh.remove_unreferenced_vertices()
                mesh.remove_non_manifold_edges()
            except Exception:
                pass

        # 计算法向量（如果需要）
        mesh.compute_vertex_normals()
        
        points = None
        normals = None
        # 固定为 Open3D 均匀采样
        try:
            pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        except Exception as oe:
            # 万一失败，降到 Poisson 再失败就抛错
            try:
                pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
            except Exception:
                raise oe

        # 获取点云数据 (x, y, z, nx, ny, nz)
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        point_cloud_data = np.concatenate([points, normals], axis=1)

        # 可选：PCA 对齐 + 原点统一 + 尺寸归一化 + 方向消歧
        if enable_pca_align and point_cloud_data.shape[1] >= 3:
            cfg = PCAAlignConfig(
                toe_positive_x=pca_toe_positive_x,
                origin_mode=pca_origin,
                heel_slice_ratio=float(pca_heel_slice_ratio),
                scale_mode=pca_scale_mode,
                target_length=float(pca_target_length),
                target_width=float(pca_target_width),
            )
            point_cloud_data, stats = pca_align_and_normalize(point_cloud_data, cfg)
            # 输出对齐信息，便于确认
            print(
                f"  PCA 对齐完成 | origin={pca_origin}, toe+X={pca_toe_positive_x}, "
                f"scale={pca_scale_mode}({stats.scale:.5f}) | bbox(Lx,Ly,Lz)={stats.length_xyz}"
            )
            print(
                f"  对齐参数 | translation(origin)={stats.translation.tolist()}, sign_flips={stats.sign_flips.tolist()}"
            )
        
        # 保存为.npy格式
        np.save(output_path, point_cloud_data)
        
        print(f"成功处理: {stl_file_path} -> {output_path}")
        print(f"原始STL - 顶点数: {original_vertices}, 面数: {original_triangles}")
        print(f"采样点云形状: {point_cloud_data.shape}")
        print("-" * 50)
        
        return True
        
    except Exception as e:
        print(f"处理文件 {stl_file_path} 时出错: {str(e)}")
        return False


def process_directory(
    input_dir,
    output_dir,
    num_points=300000,
    uniform_sampling=True,
    clean_mesh: bool = True,
    enable_pca_align: bool = True,
    pca_origin: str = "heel",
    pca_toe_positive_x: bool = True,
    pca_heel_slice_ratio: float = 0.08,
    pca_scale_mode: str = "none",
    pca_target_length: float = 1.0,
    pca_target_width: float = 1.0,
):
    """
    处理目录中的所有STL文件
    
    Args:
        input_dir (str): 输入目录路径
        output_dir (str): 输出目录路径
        num_points (int): 采样点数量
        uniform_sampling (bool): 是否使用均匀采样，默认True(使用均匀采样)
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有STL文件（不区分大小写）
    stl_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith('.stl'):
            stl_files.append(os.path.join(input_dir, file))
    
    if not stl_files:
        print(f"在目录 {input_dir} 中没有找到STL文件")
        return
    
    print(f"找到 {len(stl_files)} 个STL文件")
    
    success_count = 0
    for stl_file in stl_files:
        # 生成输出文件名
        filename = Path(stl_file).stem
        output_file = os.path.join(output_dir, f"{filename}.npy")
        
        # 处理文件
        if stl_to_pointcloud(
            stl_file,
            output_file,
            num_points=num_points,
            uniform_sampling=uniform_sampling,
            clean_mesh=clean_mesh,
            enable_pca_align=enable_pca_align,
            pca_origin=pca_origin,
            pca_toe_positive_x=pca_toe_positive_x,
            pca_heel_slice_ratio=pca_heel_slice_ratio,
            pca_scale_mode=pca_scale_mode,
            pca_target_length=pca_target_length,
            pca_target_width=pca_target_width,
        ):
            success_count += 1
    
    print(f"处理完成: {success_count}/{len(stl_files)} 个文件成功转换")


def main():
    parser = argparse.ArgumentParser(description="从STL文件中采样点云并保存为.npy格式")
    parser.add_argument("--input_dir", type=str, default="data/raw", 
                       help="输入目录路径 (默认: data/raw)")
    parser.add_argument("--output_dir", type=str, default="data/pointcloud", 
                       help="输出目录路径 (默认: data/pointcloud)")
    parser.add_argument("--num_points", type=int, default=300000, 
                       help="采样点数量 (默认: 300,000)")
    parser.add_argument("--sampling_method", type=str, default="uniform", 
                        choices=["uniform"], 
                        help="采样方法: 仅 uniform (均匀采样)")
    parser.add_argument("--subdir", type=str, choices=["feet", "insoles"], 
                       help="只处理指定的子目录")
    parser.add_argument("--main_dir_only", action="store_true", 
                       help="只处理主目录，不处理子目录")

    # 预处理（PCA/方向/原点/尺度）相关选项
    parser.add_argument("--enable_pca_align", dest="enable_pca_align", action="store_true", help="启用 PCA 对齐与归一化")
    parser.add_argument("--no_enable_pca_align", dest="enable_pca_align", action="store_false", help="禁用 PCA 对齐与归一化")
    parser.set_defaults(enable_pca_align=True)
    parser.add_argument("--pca_origin", type=str, default="heel", choices=["heel", "bottom"], help="原点统一方式")
    parser.add_argument("--pca_toe_positive_x", dest="pca_toe_positive_x", action="store_true", help="脚尖朝 +X")
    parser.add_argument("--no_pca_toe_positive_x", dest="pca_toe_positive_x", action="store_false", help="不强制脚尖朝 +X")
    parser.set_defaults(pca_toe_positive_x=True)
    parser.add_argument("--pca_heel_slice_ratio", type=float, default=0.08, help="脚跟/端部切片占总长的比例 [0, 0.3]")
    parser.add_argument("--pca_scale_mode", type=str, default="none", choices=["none", "length", "width"], help="尺度归一化模式")
    parser.add_argument("--pca_target_length", type=float, default=1.0, help="目标脚长（当 scale_mode=length）")
    parser.add_argument("--pca_target_width", type=float, default=1.0, help="目标脚宽（当 scale_mode=width）")
    
    args = parser.parse_args()
    
    # 设置采样方法
    uniform_sampling = args.sampling_method == "uniform"
    
    if args.subdir:
        # 只处理指定的子目录
        input_subdir = os.path.join(args.input_dir, args.subdir)
        output_subdir = os.path.join(args.output_dir, args.subdir)
        
        if os.path.exists(input_subdir):
            print(f"\n处理子目录: {args.subdir}")
            process_directory(
                    input_subdir,
                    output_subdir,
                    num_points=args.num_points,
                    uniform_sampling=uniform_sampling,
                    enable_pca_align=args.enable_pca_align,
                    pca_origin=args.pca_origin,
                    pca_toe_positive_x=args.pca_toe_positive_x,
                    pca_heel_slice_ratio=args.pca_heel_slice_ratio,
                    pca_scale_mode=args.pca_scale_mode,
                    pca_target_length=args.pca_target_length,
                    pca_target_width=args.pca_target_width,
                )
        else:
            print(f"子目录不存在: {input_subdir}")
    elif args.main_dir_only:
        # 只处理主目录
        process_directory(
            args.input_dir,
            args.output_dir,
            num_points=args.num_points,
            uniform_sampling=uniform_sampling,
            enable_pca_align=args.enable_pca_align,
            pca_origin=args.pca_origin,
            pca_toe_positive_x=args.pca_toe_positive_x,
            pca_heel_slice_ratio=args.pca_heel_slice_ratio,
            pca_scale_mode=args.pca_scale_mode,
            pca_target_length=args.pca_target_length,
            pca_target_width=args.pca_target_width,
        )
    else:
        # 默认处理所有子目录
        for subdir in ["feet", "insoles"]:
            input_subdir = os.path.join(args.input_dir, subdir)
            output_subdir = os.path.join(args.output_dir, subdir)
            
            if os.path.exists(input_subdir):
                print(f"\n处理子目录: {subdir}")
                process_directory(
                    input_subdir,
                    output_subdir,
                    num_points=args.num_points,
                    uniform_sampling=uniform_sampling,
                    enable_pca_align=args.enable_pca_align,
                    pca_origin=args.pca_origin,
                    pca_toe_positive_x=args.pca_toe_positive_x,
                    pca_heel_slice_ratio=args.pca_heel_slice_ratio,
                    pca_scale_mode=args.pca_scale_mode,
                    pca_target_length=args.pca_target_length,
                    pca_target_width=args.pca_target_width,
                )
            else:
                print(f"子目录不存在: {input_subdir}")


if __name__ == "__main__":
    main()
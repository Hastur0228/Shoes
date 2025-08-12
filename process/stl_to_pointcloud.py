import open3d as o3d
import numpy as np
import os
import glob
from pathlib import Path
import argparse

 


def stl_to_pointcloud(stl_file_path,
                      output_path,
                      num_points=300000,
                      uniform_sampling=True,
                      clean_mesh: bool = True):
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

        # 获取点云数据
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        
        # 组合点和法向量
        point_cloud_data = np.concatenate([points, normals], axis=1)
        
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


def process_directory(input_dir,
                      output_dir,
                      num_points=300000,
                      uniform_sampling=True,
                      clean_mesh: bool = True):
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
    
    args = parser.parse_args()
    
    # 设置采样方法
    uniform_sampling = args.sampling_method == "uniform"
    
    if args.subdir:
        # 只处理指定的子目录
        input_subdir = os.path.join(args.input_dir, args.subdir)
        output_subdir = os.path.join(args.output_dir, args.subdir)
        
        if os.path.exists(input_subdir):
            print(f"\n处理子目录: {args.subdir}")
            process_directory(input_subdir, output_subdir,
                              num_points=args.num_points,
                              uniform_sampling=uniform_sampling)
        else:
            print(f"子目录不存在: {input_subdir}")
    elif args.main_dir_only:
        # 只处理主目录
        process_directory(args.input_dir, args.output_dir,
                          num_points=args.num_points,
                          uniform_sampling=uniform_sampling)
    else:
        # 默认处理所有子目录
        for subdir in ["feet", "insoles"]:
            input_subdir = os.path.join(args.input_dir, subdir)
            output_subdir = os.path.join(args.output_dir, subdir)
            
            if os.path.exists(input_subdir):
                print(f"\n处理子目录: {subdir}")
                process_directory(input_subdir, output_subdir,
                                  num_points=args.num_points,
                                  uniform_sampling=uniform_sampling)
            else:
                print(f"子目录不存在: {input_subdir}")


if __name__ == "__main__":
    main()
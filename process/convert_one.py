import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d


def stl_to_npy_single(
    stl_path: Path,
    output_npy: Path,
    num_points: int = 200000,
    sampling_method: str = "poisson",
) -> bool:
    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if not mesh.has_vertices():
        print(f"无效STL: {stl_path}")
        return False
    try:
        mesh.compute_vertex_normals()
    except Exception:
        pass

    if sampling_method == "uniform":
        try:
            pcd = mesh.sample_points_uniformly(number_of_points=num_points, use_triangle_normal=True)
        except TypeError:
            pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    else:
        try:
            pcd = mesh.sample_points_poisson_disk(number_of_points=num_points, use_triangle_normal=True)
        except TypeError:
            pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)

    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pcd.normalize_normals()

    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals) if pcd.has_normals() else None
    if nrm is None or nrm.shape[0] != pts.shape[0]:
        nrm = np.zeros_like(pts)
    arr = np.concatenate([pts, nrm], axis=1).astype(np.float32)
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, arr)
    print(f"保存: {output_npy} | 形状: {arr.shape}")
    return True


def npy_to_stl_single(
    npy_path: Path,
    output_stl: Path,
    method: str = "poisson",
    radius: float = 0.1,
) -> bool:
    arr = np.load(npy_path)
    if arr.ndim != 2 or arr.shape[1] < 3:
        print(f"无效NPY: {npy_path}，需要 (N,>=3)")
        return False
    pts = arr[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    # 尝试使用法向
    if arr.shape[1] >= 6:
        try:
            pcd.normals = o3d.utility.Vector3dVector(arr[:, 3:6])
        except Exception:
            pass
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pcd.normalize_normals()

    if method == "ball_pivoting":
        radii = [radius, radius * 2, radius * 4]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
    elif method == "alpha_shape":
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, radius)
    else:
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9, width=0, scale=1.1, linear_fit=False
        )

    if not mesh.has_vertices() or not mesh.has_triangles():
        print("网格重建失败")
        return False

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    try:
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
    except Exception:
        pass

    output_stl.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_stl), mesh)
    print(f"保存: {output_stl} | 顶点: {len(mesh.vertices)} 面: {len(mesh.triangles)}")
    return True


def interactive_cli() -> None:
    print("=== 交互式转换 (命令行) ===  输入 q 退出")
    # 模式
    while True:
        mode = input("选择模式 [1=STL->NPY, 2=NPY->STL] (默认1): ").strip() or "1"
        if mode.lower() == "q":
            return
        if mode in {"1", "2"}:
            break
        print("无效选择，请输入 1 或 2。")

    # 输入路径
    while True:
        in_path_str = input("输入文件路径: ").strip()
        if in_path_str.lower() == "q":
            return
        if not in_path_str:
            print("必须提供输入文件")
            continue
        in_path = Path(in_path_str)
        if not in_path.exists():
            print(f"输入文件不存在: {in_path}")
            continue
        break

    # 输出路径
    default_out = in_path.with_suffix(".npy") if mode == "1" else in_path.with_suffix(".stl")
    out_path_str = input(f"输出文件路径(回车使用默认: {default_out}): ").strip()
    if out_path_str.lower() == "q":
        return
    out_path = Path(out_path_str) if out_path_str else default_out

    # 额外参数
    if mode == "1":
        num_points_in = input("采样点数(默认200000): ").strip()
        try:
            num_points = int(num_points_in) if num_points_in else 200000
        except ValueError:
            print("点数输入无效，使用默认 200000")
            num_points = 200000
        method = input("采样方法 uniform/poisson (默认poisson): ").strip() or "poisson"
        print("\n即将执行: STL -> NPY")
        print(f"输入: {in_path}")
        print(f"输出: {out_path}")
        print(f"点数: {num_points} | 采样: {method}")
        confirm = input("确认执行? [Y/n]: ").strip().lower()
        if confirm == "n":
            print("已取消")
            return
        ok = stl_to_npy_single(in_path, out_path, num_points=num_points, sampling_method=method)
    else:
        recon = input("重建方法 poisson/ball_pivoting/alpha_shape (默认poisson): ").strip() or "poisson"
        radius_in = input("重建半径/alpha (默认0.1): ").strip()
        try:
            radius = float(radius_in) if radius_in else 0.1
        except ValueError:
            print("半径/alpha 输入无效，使用默认 0.1")
            radius = 0.1
        print("\n即将执行: NPY -> STL")
        print(f"输入: {in_path}")
        print(f"输出: {out_path}")
        print(f"方法: {recon} | 半径/alpha: {radius}")
        confirm = input("确认执行? [Y/n]: ").strip().lower()
        if confirm == "n":
            print("已取消")
            return
        ok = npy_to_stl_single(in_path, out_path, method=recon, radius=radius)

    print("\n转换成功" if ok else "\n转换失败")


def launch_gui() -> None:
    print("GUI 未启用。请使用 --interactive 进入命令行交互模式。")
    return


def main():
    parser = argparse.ArgumentParser(description="单文件 STL <-> NPY 转换")
    parser.add_argument("--input", type=str, help="输入文件路径 (.stl 或 .npy)")
    parser.add_argument("--output", type=str, help="输出文件路径 (.npy 或 .stl)")
    parser.add_argument("--num_points", type=int, default=200000, help="从STL采样点数")
    parser.add_argument(
        "--sampling_method", type=str, default="poisson", choices=["uniform", "poisson"], help="STL采样方法"
    )
    parser.add_argument(
        "--recon_method", type=str, default="poisson", choices=["poisson", "ball_pivoting", "alpha_shape"], help="NPY->STL重建方法"
    )
    parser.add_argument("--radius", type=float, default=0.1, help="重建半径/alpha参数")
    parser.add_argument("--verbose", action="store_true", help="输出更多信息")
    parser.add_argument("--interactive", action="store_true", help="交互式命令行界面")
    parser.add_argument("--gui", action="store_true", help="图形界面(基于tkinter)")

    args = parser.parse_args()

    # Launch GUI
    if args.gui:
        return launch_gui()

    # Default to interactive CLI when no explicit I/O arguments are provided
    if args.interactive or (not args.input and not args.output):
        return interactive_cli()

    # Standard CLI mode (explicit arguments)
    if not args.input or not args.output:
        parser.print_help()
        return

    inp = Path(args.input)
    outp = Path(args.output)

    if not inp.exists():
        raise FileNotFoundError(f"输入文件不存在: {inp}")

    if args.verbose:
        print("=== 转换参数 ===")
        for k, v in vars(args).items():
            print(f"{k}: {v}")

    if inp.suffix.lower() == ".stl" and outp.suffix.lower() == ".npy":
        ok = stl_to_npy_single(inp, outp, num_points=args.num_points, sampling_method=args.sampling_method)
    elif inp.suffix.lower() == ".npy" and outp.suffix.lower() == ".stl":
        ok = npy_to_stl_single(inp, outp, method=args.recon_method, radius=args.radius)
    else:
        print("文件后缀不匹配：只支持 .stl->.npy 或 .npy->.stl")
        return

    if ok:
        print("转换成功")
    else:
        print("转换失败")


if __name__ == "__main__":
    main()



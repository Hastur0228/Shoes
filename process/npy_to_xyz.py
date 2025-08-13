import os
import argparse
import numpy as np
from tqdm import tqdm


def npy_to_xyz(input_dir: str, output_dir: str) -> None:
    """
    将输入目录下的所有 .npy 点云文件转换为 .xyz 文本文件，仅输出前3列 (x, y, z)。

    - 递归遍历 input_dir
    - 对每个 .npy 文件，读取为数组，取前3列作为坐标
    - 在 output_dir 中按相对路径写出 .xyz（保留子目录结构）
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    npy_files = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith('.npy'):
                npy_files.append(os.path.join(root, fname))

    if not npy_files:
        print(f"在目录 {input_dir} 中未找到 .npy 文件")
        return

    for npy_path in tqdm(npy_files, desc="转换 npy -> xyz"):
        try:
            arr = np.load(npy_path)
            if arr.ndim != 2 or arr.shape[1] < 3:
                print(f"  跳过（形状无效）: {npy_path} | shape={arr.shape}")
                continue
            points_xyz = arr[:, :3].astype(np.float64, copy=False)

            rel_path = os.path.relpath(npy_path, input_dir)
            rel_base, _ = os.path.splitext(rel_path)
            out_path = os.path.join(output_dir, rel_base + '.xyz')

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            # 写出为 ascii 文本：x y z 每行一条
            np.savetxt(out_path, points_xyz, fmt='%.8f')
        except Exception as e:
            print(f"  转换失败: {npy_path} -> {e}")

    print(f"完成：共处理 {len(npy_files)} 个 .npy 文件，输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="将 data/pointcloud 下的 .npy 点云批量转换为 .xyz")
    parser.add_argument('--input', type=str, default=os.path.join('output', 'pointcloud'),
                        help='输入根目录（默认: data/pointcloud）')
    parser.add_argument('--output', type=str, default='xyz_output',
                        help='输出根目录（默认: xyz）')
    args = parser.parse_args()

    npy_to_xyz(args.input, args.output)


if __name__ == '__main__':
    main()



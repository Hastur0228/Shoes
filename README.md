## Shoes: Foot point cloud → Insole point cloud (DGCNN)

Python 3.10+ project for training a DGCNN-based model that maps a foot point cloud to its corresponding insole point cloud. Supports large input clouds (e.g., 200k points) with memory-safe training, batch/single inference, and STL reconstruction via Open3D.

### 目录结构

```
data/
  pointcloud/
    feet/      # 训练输入，命名：001_foot_L.npy / 001_foot_R.npy ... (N, 3) 或 (N, 6: XYZ+Normal)
    insoles/   # 训练目标，命名：001_insole_L.npy / 001_insole_R.npy ... (M, 3) 或 (M, 6)
test/
  pointcloud/  # 推理输入（测试集），缺省直接放在该目录下
output/
  pointcloud/  # 推理输出（预测鞋垫点云）

process/
  dataset.py                 # 数据集与规范化（XYZ 归一化、法向单位化）
  losses.py                  # Chamfer 距离（可选下采样以避免 OOM）
  model.py                   # DGCNN + 生成头（点云→点云）
  train_point2point.py       # 训练脚本（支持左右脚过滤、200k 输入、二次采样）
  infer_point2point.py       # 推理（单文件/批量二合一，支持输出点数 target_points）
  infer_p2p_resample.py      # 方案一：小生成头 + 推理时重采样到 200k
  infer_p2p_dense.py         # 方案二：推理时强制生成头直接输出 200k
  pointcloud_to_stl.py       # 将点云转换为 STL（Open3D）
  stl_to_pointcloud.py       # 将 STL 转为点云（如需）
```

### 环境与安装

1) Python 3.10+（本项目使用 `X | Y` 联合类型语法）

2) 安装依赖（请先根据你显卡/驱动选择合适的 PyTorch CUDA 版本）

```
pip install -r requirements.txt

# 或手动安装 torch（示例：CUDA 11.8）
# pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
```

3) 可选：减少 CUDA 内存碎片（Windows PowerShell）

```
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

### 训练

- 打开 `process/train_point2point.py`，顶部 `TrainConfig` 为默认配置：
  - `num_points=300000`：数据层重采样到 30 万点
  - `encode_points=4096`：进入 DGCNN 前的二次采样，避免 KNN 和特征图显存爆炸
  - `gen_points=4096`：生成头输出较小点数（推理可再重采样到 300k）
  - `k=10`、`batch_size=1`、`num_workers=0`：大点云友好设置
  - `side_filter`: 设为 `'L'` 或 `'R'` 可实现左右脚分别训练；设为 `None` 使用两侧

运行：
```
python -m process.train_point2point
```

日志与权重：
```
checkpoints/
  p2p_dgcnn/
    run.log
    models/
      best.pt
      epoch_XXX.pt
```

### 推理（点云 → 点云）

统一入口（推荐，单/批两用）：
```
# 单文件
python -m process.infer_point2point \
  --input test/pointcloud/001_foot_L.npy \
  --output output/pointcloud/001_insole_L.npy \
  --checkpoint checkpoints/p2p_dgcnn/models/best.pt \
  --target_points 300000

# 批量（默认遍历 test/pointcloud/*.npy，输出到 output/pointcloud/）
python -m process.infer_point2point \
  --checkpoint checkpoints/p2p_dgcnn/models/best.pt \
  --target_points 300000
```

两种方案的独立脚本：

- 方案一：小生成头 + 推理阶段重采样到 300k（更省显存）
```
python -m process.infer_p2p_resample \
  --checkpoint checkpoints/p2p_dgcnn/models/best.pt \
  --encode_points 4096 \
  --target_points 300000
```

- 方案二：推理时强制生成头直接输出 300k（显存更高，速度快）
```
python -m process.infer_p2p_dense \
  --checkpoint checkpoints/p2p_dgcnn/models/best.pt \
  --encode_points 4096
```

说明：
- 推理前按训练一致的规范化：XYZ 零均值/单位球，法向单位化；输出前还原（使用保存的质心与缩放）。
- 输出点数保证：`--target_points` 会对生成的点进行精确重采样（上/下采样）到指定数量。

### 点云 → STL（可选）

```
python -m process.pointcloud_to_stl \
  output/pointcloud \
  output/raw/insoles \
  --method poisson --depth 0 --scale 1.1 --linear-fit \
  --planarize --planarize-strength 1.0
```

### 关键实现要点

- DGCNN 编码 + MLP 生成头，将全局嵌入映射为目标点云。
- `dataset.py`：
  - XYZ 归一化为零均值、单位球；法向量单位化。
  - 支持 `use_normals`（XYZ+Normal 6 维）或仅 XYZ（3 维）。
  - 支持左右脚过滤 `side_filter`（L/R）。
- `losses.py`：Chamfer 距离支持 `max_points` 下采样，适配超大点云训练（避免 O(N^2) 内存）。
- 训练稳健性：二次采样 `encode_points` 控制图构建规模；`k` 降低邻接密度；`DataParallel` 多卡可用。

### 常见问题（OOM）

- 8GB 显存建议：`encode_points≈4096`，`max_points≈4096-8192`，`k≈8-10`，`batch_size=1`。
- 设定（PowerShell）：
```
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

### 许可证

仅用于研究与内部测试。请在使用自有数据时遵守相应的隐私与合规要求。



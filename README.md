## Shoes: Foot point cloud → Insole point cloud (DGCNN)

Python 3.10+ 项目：训练与推理一个基于 DGCNN 的点云到点云模型，将足部点云映射为鞋垫点云。支持大规模输入（≥20万点）、显存友好的训练策略、单/批量推理，以及基于 Open3D 的 STL 重建与 STL↔点云工具。

### 目录结构

```
data/
  pointcloud/
    feet/       # 训练输入 (N, 3) 或 (N, 6: XYZ+Normal)
    insoles/    # 训练目标 (M, 3) 或 (M, 6)

test/
  pointcloud/   # 推理输入（默认遍历此目录）

output/
  pointcloud/   # 推理输出（预测鞋垫点云）
  raw/          # STL 输出目录（可选，重建为 .stl）

checkpoints/
  p2p_dgcnn_*/
    run.log
    models/
      best.pt / best_L.pt / best_R.pt / epoch_XXX.pt

process/
  dataset.py                 # 数据集与规范化（XYZ 零均值/单位球，法向单位化）
  losses.py                  # Chamfer 距离（可选下采样以控显存）
  model.py                   # DGCNN 编码 + 生成头（点云→点云）
  train_point2point.py       # 训练脚本（支持左右脚分侧）
  infer_point2point.py       # 推理（单/批统一入口；二次采样 + 插值到目标点数）
  infer_p2p_resample.py      # 方案一：小生成头 + 推理时重采样到高密度
  infer_p2p_dense.py         # 方案二：密集生成头 + 二次采样后插值到目标点数
  pointcloud_to_stl.py       # 点云 → STL（Open3D）
  stl_to_pointcloud.py       # STL → 点云（可选 PCA 对齐）
  npy_to_xyz.py              # 批量 NPY → ASCII XYZ 导出
  pca_align.py               # PCA 对齐与归一化工具
```

命名约定：足部输入统一为 `001_foot_L.npy` / `001_foot_R.npy`，鞋垫目标为 `001_insole_L.npy` / `001_insole_R.npy`。

### 环境与安装

1) Python 3.10+

2) 安装依赖（请先根据显卡/驱动选择合适的 PyTorch CUDA 版本）

```
pip install -r requirements.txt

# 或先安装 torch CUDA 版（示例：CUDA 11.8）
# pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
```

3) 可选（Windows PowerShell）：减少 CUDA 内存碎片

```
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

### 数据准备（可选）

若原始数据为 STL，可先转换为点云（.npy）：

```
# 将 data/raw/{feet,insoles} 下的 STL 采样为点云，输出到 data/pointcloud/{feet,insoles}
python -m process.stl_to_pointcloud \
  --input_dir data/raw \
  --output_dir data/pointcloud \
  --num_points 300000 \
  --enable_pca_align --pca_origin heel --pca_toe_positive_x
```

数据就绪后，确保：
- 训练：`data/pointcloud/feet/*.npy` 和 `data/pointcloud/insoles/*.npy` 成对存在，命名侧别一致。
- 推理：将待预测足部点云放入 `test/pointcloud/`（支持形如 `001_foot_L.npy`）。

### 训练

默认配置位于 `process/train_point2point.py` 顶部 `TrainConfig`：
- num_points=300000：数据层重采样目标
- encode_points=4096：编码前二次采样，避免 KNN 图显存飙升
- gen_points=4096：生成头输出较小点数（推理再上采样到目标密度）
- k=10, batch_size=1, num_workers=0：大点云友好设置
- side_filter：'L'/'R'/None（左右脚过滤）

运行（按侧分别训练，分别保存 best_L.pt / best_R.pt 到不同子目录）：

```
# 同时训练左右脚，分别产生 checkpoints/p2p_dgcnn_L 与 checkpoints/p2p_dgcnn_R
python -m process.train_point2point --side both --exp_name p2p_dgcnn --data_root data/pointcloud

# 仅左脚或仅右脚
python -m process.train_point2point --side L --exp_name p2p_dgcnn --data_root data/pointcloud
python -m process.train_point2point --side R --exp_name p2p_dgcnn --data_root data/pointcloud

# 不区分侧别（生成单一 best.pt；推理时可用同一模型处理两侧）
python -m process.train_point2point --side all --exp_name p2p_dgcnn --data_root data/pointcloud
```

检查点与日志：
```
checkpoints/
  p2p_dgcnn_L/ or p2p_dgcnn_R/ or p2p_dgcnn/
    run.log
    models/
      best_L.pt / best_R.pt / best.pt
      epoch_XXX.pt
```

### 推理（点云 → 点云）

统一入口（单/批两用）：`process/infer_point2point.py`

```
# 单文件
python -m process.infer_point2point \
  --input test/pointcloud/001_foot_L.npy \
  --output output/pointcloud/insoles/001_insole_L.npy \
  --checkpoint_L checkpoints/p2p_dgcnn_L/models/best_L.pt \
  --target_points 300000 \
  --secondary_points 4096

# 批量（默认遍历 test/pointcloud/*.npy，输出到 output/pointcloud/insoles/）
python -m process.infer_point2point \
  --checkpoint_L checkpoints/p2p_dgcnn_L/models/best_L.pt \
  --checkpoint_R checkpoints/p2p_dgcnn_R/models/best_R.pt \
  --test_root test/pointcloud \
  --output_root output/pointcloud \
  --output_subdir insoles \
  --target_points 300000 \
  --secondary_points 4096
```

说明：
- 推理前与训练一致的规范化：XYZ 零均值/单位球，法向单位化；输出前使用保存的质心与缩放进行还原。
- 输出点数保证：`--target_points` 会对预测点进行上/下采样（带插值）到精确目标点数。
- 若只训练了一个通用模型（`--side all`），仅传 `--checkpoint_L <best.pt>` 即可同时处理两侧；省略或留空 `--checkpoint_R`。

附：两种推理方案的独立脚本
- 方案一（推荐，显存更省）：小生成头 + 推理重采样
```
python -m process.infer_p2p_resample \
  --checkpoint_L checkpoints/p2p_dgcnn_L/models/best_L.pt \
  --checkpoint_R checkpoints/p2p_dgcnn_R/models/best_R.pt \
  --encode_points 4096 \
  --target_points 300000 \
  --recursive --include_all_npy   # 可选
```
- 方案二：密集生成头（较高显存占用，需模型训练时使用较大的 gen_points，例如 200k）
```
python -m process.infer_p2p_dense \
  --checkpoint checkpoints/p2p_dgcnn/models/best.pt \
  --encode_points 4096 \
  --secondary_points 4096
```

### 点云 → STL（可选）

将预测鞋垫点云重建为 STL（推荐目录模式）：

```
python -m process.pointcloud_to_stl \
  output/pointcloud/insoles \
  output/raw/insoles \
  --method poisson --depth 10 --scale 1.1 --linear-fit \
  --planarize --planarize-strength 1.0
```

提示：默认会进行统计离群点移除；如需关闭加上 `--no-outlier-removal`。可选 `--cluster-cleanup` 进一步清理游离簇。

### STL → 点云（可选）

对已有 STL 进行点采样（含可选 PCA 对齐/统一原点/尺度归一化）：

```
python -m process.stl_to_pointcloud \
  --input_dir data/raw \
  --output_dir data/pointcloud \
  --num_points 300000 \
  --enable_pca_align --pca_origin heel --pca_toe_positive_x
```

### 导出为 ASCII .xyz（可选）

```
python -m process.npy_to_xyz \
  --input output/pointcloud \
  --output xyz_output
```

### 关键实现要点

- DGCNN 编码 + MLP 生成头，将全局嵌入映射为目标点云。
- 训练稳健性与显存友好：
  - `encode_points` 控制图构建规模；`k` 降低邻接密度；`gen_points` 减少生成头输出点数。
  - 损失计算（Chamfer）支持对两侧随机下采样（如 ≤8192）以避免 O(N^2) 显存爆炸。
  - 多卡：如有多 GPU，自动启用 `DataParallel`。

### 常见问题（OOM）

- 8GB 显存建议：`encode_points≈4096`，`k≈8-10`，`batch_size=1`，损失下采样上限 `≈4096-8192`。
- Windows 可设置：`$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`。

### 许可证

仅用于研究与内部测试。使用自有数据时请遵守相应的隐私与合规要求。
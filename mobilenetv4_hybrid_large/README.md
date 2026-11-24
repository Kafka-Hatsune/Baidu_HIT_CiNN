# MobileNetV4_hybrid_large（PaddleX 官方分类模型）

- **来源**：PaddleX 3.7 官方模型 `MobileNetV4_hybrid_large`，图像分类，官方推理输入 **448×448**。架构来自
  `paddlex/configs/modules/image_classification/MobileNetV4_hybrid_large.yaml`。
- **推理依赖**：`paddle.inference`（PIR / `enable_cinn()`），官方权重 `inference.json`+`inference.pdiparams`。
- **训练依赖**：PaddleClas（经 PaddleX `--install`）。

单卡、FP32、关闭 TF32；设备 `metax_gpu:0`。

## 环境依赖

```bash
source /data/env.sh
paddlex --install PaddleClas --no_deps -y
pip install --no-deps opencv-python-headless==4.6.0.66 easydict faiss-cpu gast==0.3.3
```

## 推理流程（官方静态图 dynamic vs CINN）

```bash
./run_model.sh        # dynamic
./run_model_cinn.sh   # cinn, 自动生成 results/comparison.json
```

- 权重默认 `~/.paddlex/official_models/MobileNetV4_hybrid_large`（`CHECKPOINT_DIR` 可覆盖，校验 SHA256）。
- 输入 `x:[1,3,448,448]`；**官方输入必须 448×448**，用 224 会因子图 reshape 尺寸不符报错。
- CINN 生效判定：`Compiling subgraph with CINN backend`。

实测（metax_gpu:0）：dynamic 23.62 ms、cinn 22.18 ms，`max_abs_diff = 2.20e-4` ≤ 2e-3，**PASS**。

## 训练流程（dynamic vs CINN，叠加 loss 曲线）

```bash
MODE=dynamic ./train.sh
MODE=cinn    ./train.sh
python plot_loss.py --dynamic results/dynamic-train/train.log \
    --cinn results/cinn-train/train.log --out results/train --title "MobileNetV4_hybrid_large (224 crop)"
```

- 共用 `train_config.yaml`（PaddleX 生成的 PaddleClas 配置）+ 同一 `seed=42`，CINN 走 `use_cinn→backend="CINN"`。
- 默认 `EPOCHS=32`（flowers，≈512 步），逐步记 loss，`results/train/loss_curve.png` 叠加 dynamic/cinn。

实测（512 步，metax_gpu:0）：dynamic loss 4.90 → 4.62，cinn 4.90 → 4.61，两条曲线紧贴（从零训练、短程，
loss 下降幅度小但 CINN 与 dynamic 轨迹一致）。

## 注意点

- **训练用 224 裁剪，不是 448**。在 384/448 分辨率下该模型的 conv 触发 MetaX `MCDNN_STATUS_NOT_SUPPORTED`
  （`cudnn_exhaustive_search` 也无效），改 224 裁剪后可正常训练。因此 `train_config.yaml` 的
  `RandCropImage/CropImage/TimmAutoAugment` 均改为 224。**推理仍用官方 448 图**，两者不冲突。
- 原始分类配置的 `TimmAutoAugment` 为 `rand-m9-inc1-mstd1.0`，高强度会触发 PIL `posterize` 的
  `bits must be between 1 and 8` 报错；已软化为 `rand-m1-mstd0.5-inc1`（只影响增广，不影响 loss 对比）。
- 不在 METAX_GPU_WHITELIST，`train.sh` 已设 `PADDLE_PDX_DISABLE_DEV_MODEL_WL=true`。
- `results/` 下的完整产物只留本地；`train.sh` / `run_model*.sh` 跑完自动把小型对齐结果（train.log、losses.npz、train.json、inference.json、≤2 MiB 的 outputs.npz、comparison.*）归档到 `results/<platform>/` 入库。platform 自动识别（CUDA 取 GPU 型号如 `a100`，`metax_gpu`→`metax-c500`），`CINN_PLATFORM` 可覆盖。

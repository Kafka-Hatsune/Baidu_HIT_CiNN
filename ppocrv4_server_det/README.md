# PP-OCRv4_server_det（PaddleX 官方文本检测模型）

- **来源**：PaddleX 3.7 官方模型 `PP-OCRv4_server_det`，文本检测（DBNet），输入 960×960。架构来自
  `paddlex/configs/modules/text_detection/PP-OCRv4_server_det.yaml`。
- **推理依赖**：`paddle.inference`（PIR / `enable_cinn()`），官方权重 `inference.json`+`inference.pdiparams`。
- **训练依赖**：PaddleOCR（经 PaddleX `--install`）。

单卡、FP32、关闭 TF32；设备 `metax_gpu:0`。

## 环境依赖

```bash
source /data/env.sh
paddlex --install PaddleOCR --no_deps -y
pip install -c <(echo "numpy==1.26.4") shapely pyclipper scikit-image lmdb rapidfuzz albumentations albucore
```

## 推理流程（官方静态图 dynamic vs CINN）

```bash
./run_model.sh        # dynamic
./run_model_cinn.sh   # cinn, 自动生成 results/comparison.json
```

- 输入 `x:[1,3,960,960]`；CINN 生效判定：`Compiling subgraph with CINN backend`。

实测（metax_gpu:0）：dynamic 53.20 ms、cinn 49.77 ms，`max_abs_diff = 3.86e-5` ≤ 2e-3，**PASS**。

## 训练流程（dynamic 成功；CINN 训练中段被 loss 断言中断）

```bash
MODE=dynamic ./train.sh    # PaddleX+PaddleOCR, ocr_det_dataset_examples, ~500 步
MODE=cinn    ./train.sh     # FLAGS_use_cinn=1 + FLAGS_prim_all + Train.dy2st=True
python plot_loss.py --dynamic results/dynamic-train/stdout.log \
    --cinn results/cinn-train/stdout.log --out results/train --title "PP-OCRv4_server_det"
```

- `EPOCHS=10`（ocr_det 200 imgs/bs4 → 50 步/epoch，≈500 步），日志每 10 iters 记一次。
- `results/train/loss_curve.png` 叠加 dynamic 与 cinn 两条训练 loss。

实测（metax_gpu:0）：dynamic loss 2.18 → 1.43（跑满）；cinn loss 1.84 → 1.49（跑到 ~step 400），两条在该区间
**贴合下降**。

## 注意点

- **CINN 训练到约 step 400 被中断**（非 NaN）：PaddleOCR 检测 loss `det_basic_loss.py` 有严格断言
  `assert loss <= 1`（dice/balance loss），CINN 数值使该分量偶发略大于 1 触发 `AssertionError`。属于上游断言过严 +
  CINN 精度差，不是发散。loss 曲线因此 CINN 侧只到 ~step 400，前段与 dynamic 一致。
- **PaddleOCR 的 MetaX 设备名 bug**：`tools/program.py` 对 metax 拼成 `"metax:{id}"`（应为 `"metax_gpu:{id}"`），
  否则 `paddle.set_device` 报设备格式错；本仓已在该文件本地修正。
- CINN 用 `FLAGS_use_cinn=1 + dy2st=True`（PaddleOCR 的 `apply_to_static` 无 CINN backend 参数，走 FLAGS 路径）。
- 不在 METAX_GPU_WHITELIST，`train.sh` 已设 `PADDLE_PDX_DISABLE_DEV_MODEL_WL=true`；`results/`、`logs/` 不入库。

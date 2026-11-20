# PP-OCRv4_server_rec（PaddleX 官方文本识别模型）

- **来源**：PaddleX 3.7 官方模型 `PP-OCRv4_server_rec`，文本识别（CTC），输入 48×320。架构来自
  `paddlex/configs/modules/text_recognition/PP-OCRv4_server_rec.yaml`。
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

- 输入 `x:[1,3,48,320]`；CINN 生效判定：`Compiling subgraph with CINN backend`。

实测（metax_gpu:0）：dynamic 6.67 ms、cinn 7.08 ms，`max_abs_diff = 4.24e-4` ≤ 2e-3，**PASS**。

## 训练流程（dynamic vs CINN，叠加 loss 曲线）

```bash
MODE=dynamic ./train.sh    # PaddleX+PaddleOCR, ocr_rec_dataset_examples
MODE=cinn    ./train.sh     # FLAGS_use_cinn=1 + FLAGS_prim_all + Train.dy2st=True
python plot_loss.py --dynamic results/dynamic-train/stdout.log \
    --cinn results/cinn-train/stdout.log --out results/train --title "PP-OCRv4_server_rec"
```

- `EPOCHS=1`（ocr_rec 4468 imgs/bs8），日志每 10 iters 记一次。
- `results/train/loss_curve.png` 叠加 dynamic 与 cinn 两条训练 loss（CTC loss）。

实测（metax_gpu:0）：dynamic loss 21.2 → 11.7，cinn 23.8 → 10.0，两条曲线同步下降、收敛相当。**本模型 CINN 训练
全程无 NaN/断言问题**，是五个模型里 CINN 训练最顺的检测/OCR 模型。

## 注意点

- CINN 用 `FLAGS_use_cinn=1 + dy2st=True`（PaddleOCR 的 `apply_to_static` 无 CINN backend 参数，走 FLAGS 路径），
  识别模型用 CTC loss，没有检测 det/BCE loss 在 CINN 下的数值问题。
- **PaddleOCR 的 MetaX 设备名 bug**：`tools/program.py` 对 metax 拼成 `"metax:{id}"`（应为 `"metax_gpu:{id}"`），
  本仓已在该文件本地修正（与 `ppocrv4_server_det` 共用同一份 PaddleOCR）。
- 不在 METAX_GPU_WHITELIST，`train.sh` 已设 `PADDLE_PDX_DISABLE_DEV_MODEL_WL=true`；`results/`、`logs/` 不入库。

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

## 源码补丁（patch.sh）

`train.sh` 每次训练前执行 `patch.sh apply`：逐项检查 PaddleX 安装的仓库（`$PADDLEX/paddlex/repo_manager/repos`）是否已打补丁，
未打则按原文精确替换后再打上，已打则跳过；上游代码对不上时报错退出，需要手工处理。
`./patch.sh check` 只检查（未全部打上时返回 1），`./patch.sh revert` 还原成上游代码。

- `ocr_metax_device`（`PaddleOCR/tools/program.py`）：metax 设备串拼成了 `"metax:{id}"`，应为 `"metax_gpu:{id}"`，否则 `paddle.set_device` 报设备格式错误。CUDA 上不走这段代码。
- `ocr_dataset_seed`（`PaddleOCR/ppocr/data/simple_dataset.py`）：`SimpleDataSet` 构建时 `seed=None`（这个参数兼作 epoch 编号），随后 `random.seed(None)` 从系统熵重新播种，每次训练的样本顺序都不同。补丁让 `seed=None` 时改用 `CINN_TRAIN_SEED`。

种子补丁只在设置了 `CINN_TRAIN_SEED` 时生效。`train.sh` 默认 `DETERMINISTIC=1`，此时会导出 `CINN_TRAIN_SEED=${SEED:-1024}`、
`FLAGS_cudnn_deterministic=true`、`FLAGS_embedding_deterministic=1`，并传 `-o Train.log_interval=1` 逐步记 loss；
`DETERMINISTIC=0` 时不设种子，打过补丁的仓库行为与上游一致。

补丁后 A100 实测（EPOCHS=1，同配置跑两次）：dynamic 和 CINN 各自 789 个点全部逐位一致；修复前每步相对差中位数约 14%。

## 注意点

- CINN 用 `FLAGS_use_cinn=1 + dy2st=True`（PaddleOCR 的 `apply_to_static` 无 CINN backend 参数，走 FLAGS 路径），
  识别模型用 CTC loss，没有检测 det/BCE loss 在 CINN 下的数值问题。
- **PaddleOCR 的 MetaX 设备名 bug**：由 `patch.sh` 的 `ocr_metax_device` 修正（见上「源码补丁」）。
- 不在 METAX_GPU_WHITELIST，`train.sh` 已设 `PADDLE_PDX_DISABLE_DEV_MODEL_WL=true`。
- `results/` 下的完整产物只留本地；`train.sh` / `run_model*.sh` 跑完自动把小型对齐结果（train.log、losses.npz、train.json、inference.json、≤2 MiB 的 outputs.npz、comparison.*）归档到 `results/<platform>/` 入库。platform 自动识别（CUDA 取 GPU 型号如 `a100`，`metax_gpu`→`metax-c500`），`CINN_PLATFORM` 可覆盖。

# PP-YOLOE_plus-S（PaddleX 官方检测模型）

- **来源**：PaddleX 3.7 官方模型 `PP-YOLOE_plus-S`，COCO 目标检测，输入 640×640（另有 `scale_factor` 输入）。
  架构来自 `paddlex/configs/modules/object_detection/PP-YOLOE_plus-S.yaml`。
- **推理依赖**：`paddle.inference`（PIR / `enable_cinn()`），官方权重 `inference.json`+`inference.pdiparams`。
- **训练依赖**：PaddleDetection（经 PaddleX `--install`）。

单卡、FP32、关闭 TF32；设备 `metax_gpu:0`。

## 环境依赖

```bash
source /data/env.sh
paddlex --install PaddleDetection --no_deps -y
pip install -c <(echo "numpy==1.26.4") pycocotools==2.0.8 typeguard shapely terminaltables Cython pyclipper "setuptools<81" scikit-image imageio six
pip install --no-deps imgaug
```

## 推理流程（官方静态图 dynamic vs CINN）

```bash
./run_model.sh        # dynamic
./run_model_cinn.sh   # cinn, 自动生成 results/comparison.json
```

- 输入 `image:[1,3,640,640]` + `scale_factor:[1,2]`；用真实图片（`data/image_640.npz`），合成噪声会放大 dynamic/CINN 差异。
- 检测输出 `fetch_name_0=[300,6]`（class,score,x1,y1,x2,y2）本机 NMS 对并列分数行序不稳定，故比对**检测数**
  （`fetch_name_1`）+ **分数最高 50 框的排序分数分布**（atol 1e-3），不比原始行。
- CINN 生效判定：`Compiling subgraph with CINN backend`。

实测（metax_gpu:0）：检测数与排序分数分布比对 **PASS**（`fetch_name_1` 逐元素差 0.0）。

## 训练流程（dynamic 成功；CINN 训练发散，见注意点）

```bash
MODE=dynamic ./train.sh    # PaddleX+PaddleDetection, det_coco_examples, ~525 步
MODE=cinn    ./train.sh     # FLAGS_use_cinn=1 + FLAGS_prim_all + Train.dy2st=True
python plot_loss.py --dynamic results/dynamic-train/stdout.log \
    --cinn results/cinn-train/stdout.log --out results/train --title "PP-YOLOE_plus-S"
```

- `EPOCHS=3`（det_coco_examples 175 iters/epoch @bs4，≈525 步），日志每 10 iters 记一次。
- **dynamic 训练正常**：loss 4.79 → 1.44（见 `results/train/loss_curve.png`，含导出的 `best_model/inference`）。

## 注意点

- **CINN 训练在 MetaX 上数值发散为 NaN**：开 CINN 后训练首个 epoch 内 loss 变 NaN，触发
  `bce_loss_kernel.cu` 的 device-side assert（`输入须在 [0,1]`）后崩溃。去掉 `FLAGS_prim_all` 更早发散
  （step 10 即 NaN）。失败证据见 `results/cinn-train/`。因此本目录 loss 曲线**只含 dynamic**，CINN 训练标注为
  发散（不编造曲线）。PaddleClas 分类模型没有此问题，差异在检测 loss（sigmoid/BCE）在 CINN+prim 下数值不稳。
- 推理侧 CINN 正常（见上），仅**训练侧** CINN 不可用。
- PaddleDetection 依赖 `pkg_resources`（需 `setuptools`）、`imgaug`（用 `--no-deps` 装避免拉非 headless opencv）。
- 不在 METAX_GPU_WHITELIST，`train.sh` 已设 `PADDLE_PDX_DISABLE_DEV_MODEL_WL=true`；`results/`、`logs/` 不入库。

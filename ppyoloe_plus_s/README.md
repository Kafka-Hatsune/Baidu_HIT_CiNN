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

## 源码补丁（patch.sh）

`train.sh` 每次训练前执行 `patch.sh apply`：逐项检查 PaddleX 安装的仓库（`$PADDLEX/paddlex/repo_manager/repos`）是否已打补丁，
未打则按原文精确替换后再打上，已打则跳过；上游代码对不上时报错退出，需要手工处理。
`./patch.sh check` 只检查（未全部打上时返回 1），`./patch.sh revert` 还原成上游代码。

- `ppdet_seed`（`PaddleDetection/tools/train.py`）：PaddleDetection 只有在传 `--enable_ce` 时才固定种子，而 PaddleX 只在 benchmark 模式下才传，正常训练时什么都不播种：新的 4 类分类头随机初始化，增强也不固定，第 0 步（lr=0）的 loss 都不一样。补丁在没传 `--enable_ce` 时改用 `CINN_TRAIN_SEED` 调 `set_random_seed`。

种子补丁只在设置了 `CINN_TRAIN_SEED` 时生效。`train.sh` 默认 `DETERMINISTIC=1`，此时会导出 `CINN_TRAIN_SEED=${SEED:-1024}`、
`FLAGS_cudnn_deterministic=true`、`FLAGS_embedding_deterministic=1`，并传 `-o Train.log_interval=1` 逐步记 loss；
`DETERMINISTIC=0` 时不设种子，打过补丁的仓库行为与上游一致。

补丁后 A100 实测（EPOCHS=1，同配置跑两次）：前 2 步逐位一致，之后受 GPU 原子加反向（插值、gather、gather_nd、take_along_axis、bce）影响分叉，标签分配器会放大差异，每步相对差中位数仍有 3% 至 5%（两轮实测）。注意不要同时跑两个 YOLO 训练，它们会抢写 PaddleX 目录下的 `bbox.json`。

## 注意点

- **CINN 训练在 MetaX 上数值发散为 NaN**：开 CINN 后训练首个 epoch 内 loss 变 NaN，触发
  `bce_loss_kernel.cu` 的 device-side assert（`输入须在 [0,1]`）后崩溃。去掉 `FLAGS_prim_all` 更早发散
  （step 10 即 NaN）。失败证据见 `results/cinn-train/`。因此本目录 loss 曲线**只含 dynamic**，CINN 训练标注为
  发散（不编造曲线）。PaddleClas 分类模型没有此问题，差异在检测 loss（sigmoid/BCE）在 CINN+prim 下数值不稳。
- 推理侧 CINN 正常（见上），仅**训练侧** CINN 不可用。
- PaddleDetection 依赖 `pkg_resources`（需 `setuptools`）、`imgaug`（用 `--no-deps` 装避免拉非 headless opencv）。
- 不在 METAX_GPU_WHITELIST，`train.sh` 已设 `PADDLE_PDX_DISABLE_DEV_MODEL_WL=true`。
- `results/` 下的完整产物只留本地；`train.sh` / `run_model*.sh` 跑完自动把小型对齐结果（train.log、losses.npz、train.json、inference.json、≤2 MiB 的 outputs.npz、comparison.*）归档到 `results/<platform>/` 入库。platform 自动识别（CUDA 取 GPU 型号如 `a100`，`metax_gpu`→`metax-c500`），`CINN_PLATFORM` 可覆盖。

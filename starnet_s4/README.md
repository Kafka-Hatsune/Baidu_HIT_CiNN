# StarNet-S4（PaddleX 官方分类模型）

- **来源**：PaddleX 3.7 官方模型 `StarNet-S4`，图像分类，输入 224×224。架构来自 PaddleX 自带配置
  `paddlex/configs/modules/image_classification/StarNet-S4.yaml` → 底层 `PaddleClas_api/configs/StarNet-S4.yaml`。
- **推理依赖**：`paddle.inference`（PIR 执行器 / `enable_cinn()`），官方权重 `inference.json`+`inference.pdiparams`。
- **训练依赖**：PaddleClas（经 PaddleX `--install`）。

单卡、FP32、关闭 TF32；设备 `metax_gpu:0`。`source /data/env.sh` 后运行。

## 环境依赖

```bash
source /data/env.sh                         # metax conda, Paddle 3.5.0.dev, is_compiled_with_cinn()==True
paddlex --install PaddleClas --no_deps -y   # 关键: --no_deps 保护 paddlepaddle-metax 不被 CPU 版覆盖
pip install --no-deps opencv-python-headless==4.6.0.66 easydict faiss-cpu gast==0.3.3
```

详见顶层 `README.md` 的「框架依赖安装注意点」。

## 推理流程（官方静态图 dynamic vs CINN）

PaddleX 发布的是推理图，这里比较**同一张 PIR 静态图**在默认执行器 vs `enable_cinn()` 下的结果与耗时。

```bash
./run_model.sh        # dynamic
./run_model_cinn.sh   # cinn
# 产物 results/<mode>-infer/{inference.json,outputs.npz}, 第二个模式跑完自动生成 results/comparison.json
```

- 权重默认取 `~/.paddlex/official_models/StarNet-S4`（`CHECKPOINT_DIR` 可覆盖，入口校验 SHA256）。
- 输入固定种子合成张量 `x:[1,3,224,224]`；`INPUTS_NPZ` 可复用另一平台输入做跨平台比对。
- CINN 生效判定：日志须出现 `Compiling subgraph with CINN backend`。

实测（metax_gpu:0）：dynamic 16.82 ms、cinn 16.68 ms，`max_abs_diff = 4.75e-5` ≤ 2e-3，**PASS**。

## 训练流程（dynamic vs CINN，叠加 loss 曲线）

PaddleX 框架可训练该模型（PaddleClas 后端）。两种模式共用 `train_config.yaml`（PaddleX 生成的 resolved
PaddleClas 配置）与同一 `seed`，只差执行方式，便于叠加对比。

```bash
# flowers 示例集(102类): curl -O .../paddlex/data/cls_flowers_examples.tar 到 /data/paddlex_train/datasets/
MODE=dynamic ./train.sh    # 纯动态图
MODE=cinn    ./train.sh     # Global.use_cinn=True -> to_static(backend="CINN")
python plot_loss.py --dynamic results/dynamic-train/train.log \
    --cinn results/cinn-train/train.log --out results/train --title "StarNet-S4"
```

- 默认 `EPOCHS=32`（flowers 1020/bs64 → 16 步/epoch，≈512 步）、`warmup=5`、`seed=42`、逐步记 loss。
- `plot_loss.py` 把 dynamic 与 cinn 两条训练 loss 叠加到一张图（无 lr 轴）：`results/train/loss_curve.png`。
- CINN 训练走 PaddleClas 的 `use_cinn→backend="CINN"` 干净路径（自动开 prim），非手动 FLAGS。

实测（512 步，metax_gpu:0）：dynamic loss 5.04 → 2.22，cinn 5.04 → 1.93，两条曲线贴合；CINN 稳态比 dynamic
更快。训练末自动 export 推理图，可直接接 `infer_impl.py`（见 `results/trained-*`，trained-dynamic vs
trained-cinn `max_abs_diff = 2.67e-5` PASS）。

## 注意点

- `StarNet-S4` 不在 `paddlex/utils/custom_device_list.py:METAX_GPU_WHITELIST`，`train.sh` 已内置
  `PADDLE_PDX_DISABLE_DEV_MODEL_WL=true` 跳过设备白名单断言。
- 可复现训练：`train.sh` 默认 `DETERMINISTIC=1`，导出 `FLAGS_cudnn_deterministic=true`。不开时卷积反向会选非确定性算法，同配置两次训练从第 2～3 步开始分叉（A100 实测 StarNet 5 次 dynamic 的最后 50 点均值在 1.79–2.99 之间）；开启后 dynamic、CINN 重跑均逐位一致。`DETERMINISTIC=0` 恢复上游行为。
- `results/` 下的完整产物只留本地；`train.sh` / `run_model*.sh` 跑完自动把小型对齐结果（train.log、losses.npz、train.json、inference.json、≤2 MiB 的 outputs.npz、comparison.*）归档到 `results/<platform>/` 入库。platform 自动识别（CUDA 取 GPU 型号如 `a100`，`metax_gpu`→`metax-c500`），`CINN_PLATFORM` 可覆盖；已存在的 `results/<mode>-<task>` 不被覆盖。
- 跨平台：`infer_impl.py` 与 `train.sh` 设备无关（CUDA→`gpu`，否则首个自定义设备），`PADDLE_DEVICE` 可强制；`PADDLEX`、`DATASET_DIR`/`DATASET_ROOT`、`GPU_ID`、`RESULTS_DIR` 可覆盖默认路径。

# PaddleFormers CINN Model Test Suite

这是一个按模型组织的 PaddleFormers 动态图/CINN 测试仓。每个模型目录提供统一的运行入口和模型说明；其他文件由模型实现按需组织，不作为跨模型的固定模板。

本仓库特别针对Metax平台（C500）和Ampere平台（A100）提供统一的资源记录和运行入口。

## 模型目录

### PaddleFormers（动态图 `nn.Layer` + `to_static(backend="CINN")`，含训练入口）

- [Gemma3](gemma3/run_model.md)
- [Llama4 text tiny](llama4/run_model.md)：FP32 框架测试，权重仅本地提供。
- [Llama Guard4 图文 tiny](llama_guard4/run_model.md)
- [DeepSeek-OCR2](deepseek_ocr2/run_model.md)
- [FastVLM](fastvlm/run_model.md)
- [LFM2-VL](lfm2_vl/run_model.md)
- [MiniMax](minimax/run_model.md)
- [Mistral3-Pixtral](mistral3_pixtral/run_model.md)
- [Molmo](molmo/run_model.md)
- [OLMo2](olmo2/run_model.md)
- [Phi4-Multimodal](phi4_multimodal/run_model.md)

### PaddleClas（PaddleClas backbone + `to_static(backend="CINN")`，含训练入口）

- [GhostNetV3-x1.0](ghostnetv3/run_model.md)
- [MobileNetV5](mobilenetv5/run_model.md)
- [NaFlexViT](naflexvit/run_model.md)
- [YOLO11-cls-n](yolo11/run_model.md)

### PaddleX 3.7 官方推理模型（静态 PIR 图，`AnalysisConfig.enable_cinn()`，仅推理）

PaddleX 只发布 `inference.json` + `inference.pdiparams`，没有动态图代码，因此对比的是同一张静态图在默认 PIR
执行器与 CINN 执行器下的运行，没有 `train.sh`。

- [StarNet-S4](starnet_s4/run_model.md)：图像分类，输入 224×224。
- [MobileNetV4_hybrid_large](mobilenetv4_hybrid_large/run_model.md)：图像分类，输入 448×448。
- [PP-YOLOE_plus-S](ppyoloe_plus_s/run_model.md)：目标检测，输入 640×640（另有 `scale_factor` 输入）。
- [PP-OCRv4_server_det](ppocrv4_server_det/run_model.md)：文本检测，输入 960×960。
- [PP-OCRv4_server_rec](ppocrv4_server_rec/run_model.md)：文本识别，输入 48×320。

### PaddleScience（上游 CINN 用例 + `to_static(backend="CINN")`，含训练入口）

模型与合成批次取自 `$PaddleScience/test/cinn/cases`，不复制架构。

- [darcy](darcy/run_model.md)：2D Darcy flow，MLP PINN。
- [xpinn](xpinn/run_model.md)：2D Poisson，3 子域 XPINN。
- [earthformer](earthformer/run_model.md)：ENSO CuboidTransformer。

## 运行条件

所有入口都支持 `GPU_ID`（默认 0）和 `RESULTS_DIR`（默认 `<模型目录>/results`），拒绝覆盖已存在的结果目录；
`ITERS` / `WARMUP` 在 PaddleX、PaddleScience 及部分 PaddleFormers 目录中生效（以各目录 `run_model.md` 为准）。
CINN 模式的日志必须出现 `Compiling subgraph with CINN backend`，否则退出码 4。
本机（Metax C500）先执行 `source /data/env.sh`（metax conda + Paddle 3.5.0.dev，已编入 CINN）；A100 侧换成带
CINN 的 CUDA Paddle 环境即可，设备默认值 `auto` 会选择 `gpu:0`。

### PaddleFormers

需要 PaddleFormers 源码和本地权重：

```bash
export PADDLEFORMERS_DIR=/path/to/PaddleFormers   # HEAD 必须等于目录下的 paddleformers.commit
export CHECKPOINT_DIR=/path/to/local-weights      # 见各目录 checkpoint/README.md
./run_model.sh        # dynamic
./run_model_cinn.sh   # CINN
```

入口会校验 `PADDLEFORMERS_DIR` 的 commit 与 `paddleformers.commit` 一致；`train.sh` / `infer.sh` 提供单独的
训/推入口。Gemma3 未设 `PADDLEFORMERS_DIR` 时默认取 `../PaddleFormers`。

### PaddleClas

```bash
export PADDLECLAS_DIR=/path/to/PaddleClas         # HEAD 必须等于 paddleclas.commit
export CHECKPOINT_DIR=/path/to/local-weights
./run_model.sh
./run_model_cinn.sh
```

GhostNetV3 使用的历史提交需要先应用 `ghostnetv3/patches/import-syntax.patch`（只修正导入语法），入口会严格
检查工作树只包含这份补丁；其余目录要求干净工作树。当前这些目录的 `infer_impl.py` 固定
`paddle.set_device("gpu:0")`，即面向 A100/CUDA，MetaX 上需按“跨平台”一节改成自定义设备。

### PaddleX

保证paddlex已在当前 Python 环境中安装(`import paddlex`).

PaddleX 的 `PaddleStaticRunner` 只在 `gpu`/`dcu` 分支里打开 CINN，`metax_gpu` 分支没有接，所以这些目录直接配置 `paddle.inference`。可选 `PADDLE_DEVICE`、`INPUTS_NPZ`（复用另一平台的输入）。

```bash
# 默认 $PADDLEX_MODELS_DIR/<模型名>，未设置时为 ~/.paddlex/official_models/<模型名>
export CHECKPOINT_DIR=/path/to/<模型名>   # 可选，覆盖默认目录；入口校验 3 个文件的 SHA256
./run_model.sh        # dynamic：默认 PIR 执行器
./run_model_cinn.sh   # cinn：AnalysisConfig.enable_cinn()
```

### PaddleScience

- 需要导出当前PaddleScience的位置到环境变量, 并保证已安装
- 测试使用 PaddleScience HEAD = `34819214b3f616b1112012a71131193116301283`

```bash
export PADDLE_PSCI_DIR=/data/PaddleScience
./run_model.sh        # dynamic
./run_model_cinn.sh   # cinn
```

不使用外部权重：dynamic 在 CPU 上按固定种子初始化模型与批次并写入
`results/dynamic-infer/{init.pdparams,inputs.npz}`，cinn 自动复用；`CHECKPOINT_DIR` / `INPUTS_NPZ` 可指定另一
平台导出的同一份权重/输入，`inference.json` 的 `weights_sha256` / `inputs_sha256` 用于核对。

本目录同时提供推理和训练入口；训练复用上游 case 的 `compute_loss`（Adam、lr 与用例一致、dropout 置 0），
默认 `WARMUP=5`、`ITERS=500`（总步数 505，前 5 步不计时），权重与批次同样在 CPU 上按固定种子生成：

```bash
./train.sh              # dynamic
MODE=cinn ./train.sh    # CINN，自动复用 dynamic-train 的 init.pdparams / batch.npz
```

产物：`results/{dynamic,cinn}-train/`（`train.json`、`losses.npz`、`init.pdparams`、`batch.npz`、`final.pdparams`），
比对写入 `results/comparison-train.json`，口径为 400-500 窗口、tol 1e-2、最多 10 个离群步（与上游 harness 一致）。
跨平台时用 `CHECKPOINT_DIR`（init.pdparams）+ `BATCH_NPZ`（batch.npz 及其同名 `.json`）复用另一端的权重与批次，
`train.json` 的 `init_weights_sha256` / `batch_sha256` 可确认两端同源。

2026-11-20 在 MetaX 上实测（默认 5/500）：

- `darcy`：PASS，窗口最大差 1.24e-2、1 个离群步，训练 38.96 → 18.51 ms/step（CINN 2.1×）。
- `xpinn`：PASS，窗口最大差 2.74e-5、0 个离群步，训练 120.55 → 95.86 ms/step（CINN 1.26×）。
- `earthformer`：PASS（窗口最大相对差 1.28e-1、9 个离群步，接近 10 的上限），训练 399.68 → 173.15 ms/step（CINN 2.31×）。

## 跨平台（A100 / Metax C500）

设备选择现状（以各目录代码为准）：

- **设备无关**：PaddleX 5 个目录、PaddleScience 3 个目录（CUDA 构建自动用 `gpu:0`，否则用首个自定义设备
  `metax_gpu:0`，`PADDLE_DEVICE` 可强制）；`llama4`、`llama_guard4` 同样自动选择；`gemma3` 通过 `--device` 指定。
- **当前固定 `gpu:0`（A100/CUDA）**：`deepseek_ocr2`、`fastvlm`、`lfm2_vl`、`minimax`、`mistral3_pixtral`、
  `molmo`、`olmo2`、`phi4_multimodal` 以及 PaddleClas 的 `ghostnetv3`、`mobilenetv5`、`naflexvit`、`yolo11`。
  在 MetaX 上运行需要先把 `paddle.set_device("gpu:0")` 换成自定义设备（或按 llama4 的方式自动选择）。

跨平台比对：

- PaddleX 用 `INPUTS_NPZ` 复用输入；PaddleScience 推理用 `CHECKPOINT_DIR` + `INPUTS_NPZ`，训练用
  `CHECKPOINT_DIR` + `BATCH_NPZ` 复用权重与批次。
- `python compare.py --dynamic <平台A/dynamic-infer> --cinn <平台B/dynamic-infer> --report ...`，
  差异按各目录 `model_spec.json` 的 `atol` 判定（不同平台浮点实现不同）。

## 主要交付物说明

| 文件或目录 | 功能 |
| --- | --- |
| `run_model.sh` | dynamic 快速入口。 |
| `run_model_cinn.sh` | CINN 快速入口。 |
| `train.sh` | 单独训练入口（PaddleFormers / PaddleClas / PaddleScience 目录提供）。 |
| `infer.sh` | 单独推理入口。 |
| `run_model.py` / `infer_impl.py` / `train_impl.py` | 构建模型/批次、执行推理或训练、记录性能与结果。 |
| `compare.py` | dynamic 与 CINN 输出比对（PaddleX / PaddleScience 目录）。 |
| `model_spec.json` | 输入规格、种子、比对阈值（PaddleX / PaddleScience 目录）。 |
| `run_model.md` | 模型专属环境、数据、运行方式和结果说明。 |
| `checkpoint/` | 权重说明或外部存放位置记录；权重不入库。 |
| `results/`、`logs/` | 运行产物，默认只保存在本地。 |

- 每个模型应能从自己的 `run_model.md` 中找到入口、环境变量和结果查看方式。
- `logs/` 和 `results/` 中的产物默认不提交；已存在的结果目录不会被覆盖。
- PaddleX 目录没有 `train.sh`（只发布静态推理图）；PaddleScience 的 `train.sh` 复用上游 case 的
  `compute_loss`，默认 `WARMUP=5`、`ITERS=500`。

## 验收记录

- PaddleX 5 个模型：2026-11-20 在 `metax_gpu:0` 上 dynamic/CINN 推理均通过，数值差异见各目录 `run_model.md`。
- PaddleScience 3 个用例：推理 dynamic/CINN 均通过；训练（默认 `WARMUP=5`、`ITERS=500`）`darcy`、`xpinn`、
  `earthformer` 的实测结果见各目录 `run_model.md`。
- PaddleFormers / PaddleClas 各目录的验收结果记录在各自的 `run_model.md` 中。

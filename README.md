# PaddleFormers CINN Model Test Suite

按模型组织的动态图 / CINN 对照测试仓，覆盖 PaddleFormers、PaddleClas、PaddleX、PaddleScience 四类来源。
每个模型目录提供统一的运行入口（推理，部分含训练）与模型说明（`README.md` 或 `run_model.md`）；特别针对
Metax 平台（C500）与 Ampere 平台（A100）提供统一的资源记录和运行入口。

## 模型目录

以**模型**为单位，标注「来源 / 推理依赖 / 训练依赖框架」。入口文档见各链接。

- [Gemma3](gemma3/run_model.md) — 来源：PaddleFormers Gemma3；推理依赖：PaddleFormers（`nn.Layer`+`to_static(backend="CINN")`）；训练依赖：PaddleFormers。
- [Llama4 text tiny](llama4/run_model.md) — 来源：PaddleFormers Llama4（FP32，权重本地）；推理依赖：PaddleFormers；训练依赖：PaddleFormers。
- [Llama Guard4 图文 tiny](llama_guard4/run_model.md) — 来源：PaddleFormers LlamaGuard4；推理依赖：PaddleFormers；训练依赖：PaddleFormers。
- [DeepSeek-OCR2](deepseek_ocr2/run_model.md) — 来源：PaddleFormers DeepSeek-OCR2；推理依赖：PaddleFormers；训练依赖：PaddleFormers。
- [FastVLM](fastvlm/run_model.md) — 来源：PaddleFormers FastVLM；推理依赖：PaddleFormers；训练依赖：PaddleFormers。
- [LFM2-VL](lfm2_vl/run_model.md) — 来源：PaddleFormers LFM2-VL；推理依赖：PaddleFormers；训练依赖：PaddleFormers。
- [MiniMax](minimax/run_model.md) — 来源：PaddleFormers MiniMax；推理依赖：PaddleFormers；训练依赖：PaddleFormers。
- [Mistral3-Pixtral](mistral3_pixtral/run_model.md) — 来源：PaddleFormers Mistral3-Pixtral；推理依赖：PaddleFormers；训练依赖：PaddleFormers。
- [Molmo](molmo/run_model.md) — 来源：PaddleFormers Molmo；推理依赖：PaddleFormers；训练依赖：PaddleFormers。
- [OLMo2](olmo2/run_model.md) — 来源：PaddleFormers OLMo2；推理依赖：PaddleFormers；训练依赖：PaddleFormers。
- [Phi4-Multimodal](phi4_multimodal/run_model.md) — 来源：PaddleFormers Phi4-Multimodal；推理依赖：PaddleFormers；训练依赖：PaddleFormers。
- [GhostNetV3-x1.0](ghostnetv3/run_model.md) — 来源：PaddleClas backbone；推理依赖：PaddleClas（`to_static(backend="CINN")`）；训练依赖：PaddleClas。
- [MobileNetV5](mobilenetv5/run_model.md) — 来源：PaddleClas backbone；推理依赖：PaddleClas；训练依赖：PaddleClas。
- [NaFlexViT](naflexvit/run_model.md) — 来源：PaddleClas backbone；推理依赖：PaddleClas；训练依赖：PaddleClas。
- [YOLO11-cls-n](yolo11/run_model.md) — 来源：PaddleClas backbone；推理依赖：PaddleClas；训练依赖：PaddleClas。
- [StarNet-S4](starnet_s4/README.md) — 来源：PaddleX 3.7 官方分类（224×224）；推理依赖：`paddle.inference`（PIR/CINN）；训练依赖：PaddleClas。
- [MobileNetV4_hybrid_large](mobilenetv4_hybrid_large/README.md) — 来源：PaddleX 3.7 官方分类（448×448）；推理依赖：`paddle.inference`（PIR/CINN）；训练依赖：PaddleClas（训练用 224 裁剪）。
- [PP-YOLOE_plus-S](ppyoloe_plus_s/README.md) — 来源：PaddleX 3.7 官方检测（640×640 + `scale_factor`）；推理依赖：`paddle.inference`（PIR/CINN）；训练依赖：PaddleDetection（CINN 训练发散，仅 dynamic）。
- [PP-OCRv4_server_det](ppocrv4_server_det/README.md) — 来源：PaddleX 3.7 官方文本检测（960×960）；推理依赖：`paddle.inference`（PIR/CINN）；训练依赖：PaddleOCR（CINN 训练中段被 loss 断言中断）。
- [PP-OCRv4_server_rec](ppocrv4_server_rec/README.md) — 来源：PaddleX 3.7 官方文本识别（48×320）；推理依赖：`paddle.inference`（PIR/CINN）；训练依赖：PaddleOCR。
- [darcy](darcy/run_model.md) — 来源：PaddleScience `test/cinn/cases`（2D Darcy, MLP PINN）；推理依赖：PaddleScience（`to_static(backend="CINN")`）；训练依赖：PaddleScience。
- [xpinn](xpinn/run_model.md) — 来源：PaddleScience `test/cinn/cases`（2D Poisson, 3 子域 XPINN）；推理依赖：PaddleScience；训练依赖：PaddleScience。
- [earthformer](earthformer/run_model.md) — 来源：PaddleScience `test/cinn/cases`（ENSO CuboidTransformer）；推理依赖：PaddleScience；训练依赖：PaddleScience。

## 框架依赖安装注意点

通用：本机先 `source /data/env.sh`（metax conda，Paddle `3.5.0.dev20260516`，`is_compiled_with_cinn()==True`）。
核心红线是**绝不让 pip 把 `paddlepaddle-metax` 覆盖成 CPU 版**，恢复 wheel 在
`/data/paddlepaddle_metax-3.5.0.dev20260516+maca3.8.4.9-cp312-cp312-linux_x86_64.whl`。

### PaddleX 训练后端（PaddleClas / PaddleDetection / PaddleOCR）

- 一律用 `paddlex --install <Repo> --no_deps -y` 安装：`--no_deps` 避免其 requirements 里的 `paddlepaddle` 覆盖 metax 版。
- 漏装的轻量依赖再单独补，且 opencv 必须 headless，否则 `import cv2` 报 `libGL.so.1`：
  - PaddleClas：`pip install --no-deps opencv-python-headless==4.6.0.66 easydict faiss-cpu gast==0.3.3`
  - PaddleDetection：`pip install pycocotools==2.0.8 typeguard shapely terminaltables Cython pyclipper "setuptools<81" scikit-image imageio six`（约束 `numpy==1.26.4`）+ `pip install --no-deps imgaug`。缺 `setuptools` 会报 `No module named pkg_resources`。
  - PaddleOCR：`pip install shapely pyclipper scikit-image lmdb rapidfuzz albumentations albucore`（约束 `numpy==1.26.4`；albumentations 会把 opencv-headless 升到 4.11，不影响）。
- **设备白名单**：这 5 个模型不全在 `paddlex/utils/custom_device_list.py:METAX_GPU_WHITELIST`，训练需
  `export PADDLE_PDX_DISABLE_DEV_MODEL_WL=true`（各 `train.sh` 已内置）。
- **PaddleOCR 设备名 bug**：`PaddleOCR/tools/program.py` 对 metax 拼成 `"metax:{id}"`（应 `"metax_gpu:{id}"`），由 OCR 目录的 `patch.sh` 修正。
- **训练可复现补丁**：det/rec/PP-YOLOE 的 `train.sh` 每次训练前跑 `patch.sh apply`，检查并补上 PaddleOCR / PaddleDetection 的种子补丁（原因见各目录 README「源码补丁」）。
- **CINN 训练能力差异**：PaddleClas 走 `Global.use_cinn=True → to_static(backend="CINN")` 干净路径；
  PaddleDetection/PaddleOCR 的 `apply_to_static` 无 CINN backend 参数，需 `FLAGS_use_cinn=1 + FLAGS_prim_all=true + Train.dy2st=True`。
  检测类 loss（BCE/dice）在 CINN+prim 下数值不稳：PP-YOLOE 训练 NaN、PP-OCR det 触发 `assert loss<=1`；OCR rec（CTC）与分类模型正常。

### PaddleFormers / PaddleClas（独立仓）/ PaddleScience

- 这三类沿用各自仓库源码 + 环境变量（`PADDLEFORMERS_DIR` / `PADDLECLAS_DIR` / `PADDLE_PSCI_DIR`），见下「运行条件」。
## 运行条件

所有入口支持 `GPU_ID`（默认 0）与 `RESULTS_DIR`（默认 `<模型目录>/results`），拒绝覆盖已存在的结果目录；
`ITERS` / `WARMUP` 在 PaddleX、PaddleScience 及部分 PaddleFormers 目录生效（以各目录文档为准）。CINN 模式日志
必须出现 `Compiling subgraph with CINN backend`，否则退出码 4。A100 侧换成带 CINN 的 CUDA Paddle，设备默认
`auto` 选 `gpu:0`。

### PaddleFormers

```bash
export PADDLEFORMERS_DIR=/path/to/PaddleFormers   # HEAD 必须等于目录下 paddleformers.commit
export CHECKPOINT_DIR=/path/to/local-weights      # 见各目录 checkpoint/README.md
./run_model.sh        # dynamic
./run_model_cinn.sh   # CINN
```

入口校验 `PADDLEFORMERS_DIR` 的 commit；`train.sh`/`infer.sh` 为单独训/推入口。Gemma3 未设时默认取 `../PaddleFormers`。

### PaddleClas（独立仓）

```bash
export PADDLECLAS_DIR=/path/to/PaddleClas         # HEAD 必须等于 paddleclas.commit
export CHECKPOINT_DIR=/path/to/local-weights
./run_model.sh ; ./run_model_cinn.sh
```

GhostNetV3 需先应用 `ghostnetv3/patches/import-syntax.patch`。这些目录的 `infer_impl.py` 固定 `gpu:0`（面向 A100），
MetaX 上需按「跨平台」改成自定义设备。

### PaddleX（含训练）

```bash
# 推理：默认 ~/.paddlex/official_models/<模型名>，CHECKPOINT_DIR 可覆盖并校验 SHA256
./run_model.sh ; ./run_model_cinn.sh        # dynamic / cinn（AnalysisConfig.enable_cinn）
# 训练：见各目录 README.md（框架依赖见上「安装注意点」）
MODE=dynamic ./train.sh ; MODE=cinn ./train.sh
python plot_loss.py --dynamic results/dynamic-train/*.log --cinn results/cinn-train/*.log --out results/train --title <模型>
```

PaddleX `PaddleStaticRunner` 只在 `gpu`/`dcu` 分支开 CINN，`metax_gpu` 没接，故推理直接配 `paddle.inference`。

PaddleX 5 目录的 `train.sh`/`run_model*.sh` 跑完会用 `archive.py` 把小型对齐结果归档到 `results/<platform>/`
（自动识别 `a100`/`metax-c500`，`CINN_PLATFORM` 覆盖）；`.gitignore` 只放行 `results/a100`、`results/metax-c500`，新平台需加白名单。

### PaddleScience

```bash
export PADDLE_PSCI_DIR=/data/PaddleScience          # HEAD=34819214b3f616b1112012a71131193116301283
./run_model.sh ; ./run_model_cinn.sh               # 推理
./train.sh ; MODE=cinn ./train.sh                  # 训练，默认 WARMUP=5 ITERS=500
```

权重/批次在 CPU 按固定种子生成并落盘，cinn 自动复用；跨平台用 `CHECKPOINT_DIR`+`INPUTS_NPZ`/`BATCH_NPZ`。
## 跨平台（A100 / Metax C500）

设备选择现状（以各目录代码为准）：

- **设备无关**：PaddleX 5 个目录、PaddleScience 3 个目录（CUDA→`gpu:0`，否则首个自定义设备 `metax_gpu:0`，
  `PADDLE_DEVICE` 可强制）；`llama4`、`llama_guard4` 同样自动选择；`gemma3` 用 `--device`。
- **当前固定 `gpu:0`（A100/CUDA）**：`deepseek_ocr2`、`fastvlm`、`lfm2_vl`、`minimax`、`mistral3_pixtral`、
  `molmo`、`olmo2`、`phi4_multimodal` 及 PaddleClas 的 `ghostnetv3`、`mobilenetv5`、`naflexvit`、`yolo11`，
  MetaX 上需改成自定义设备。

比对：PaddleX 用 `INPUTS_NPZ` 复用输入；PaddleScience 推理用 `CHECKPOINT_DIR`+`INPUTS_NPZ`、训练用
`CHECKPOINT_DIR`+`BATCH_NPZ`。`python compare.py --dynamic <A/dynamic-infer> --cinn <B/dynamic-infer>`，差异按各
目录 `model_spec.json` 的 `atol` 判定。

## 主要交付物说明

- `run_model.sh` / `run_model_cinn.sh`：dynamic / CINN 推理入口。
- `train.sh`：训练入口（PaddleFormers / PaddleClas / PaddleScience / **PaddleX 5 目录**；PaddleX 用 `MODE=dynamic|cinn`）。
- `plot_loss.py`：把 dynamic 与 CINN 两条训练 loss 叠加成一张图（PaddleX 目录，无 lr 轴）。
- `infer.sh` / `run_model.py` / `infer_impl.py` / `train_impl.py`：构建模型/批次、执行推理或训练、记录结果。
- `compare.py` / `model_spec.json`：输出比对与输入规格/阈值（PaddleX / PaddleScience）。
- `README.md`（PaddleX 5 目录）/ `run_model.md`（其余）：模型专属环境、训推流程与结果。
- `checkpoint/`：权重说明/外部位置；`results/`、`logs/`：运行产物，默认不入库、不覆盖已有目录。

## 验收记录

- PaddleX 推理：5 个模型 2026-11-20 在 `metax_gpu:0` 上 dynamic/CINN 均 PASS（差异见各目录 `README.md`）。
- PaddleX 训练（2026-11-20，`metax_gpu:0`，dynamic vs CINN loss 叠加，详见各目录 `README.md`）：
  - StarNet-S4：dynamic/CINN 均跑满 512 步，loss 5.04→2.22 / 5.04→1.93，曲线贴合。
  - MobileNetV4_hybrid_large：512 步（**训练用 224 裁剪**，384/448 触发 MCDNN 不支持），loss 4.90→4.62 / 4.90→4.61。
  - PP-OCRv4_server_rec：dynamic/CINN 同步下降 21.2→11.7 / 23.8→10.0，CINN 训练无异常。
  - PP-OCRv4_server_det：dynamic 跑满（2.18→1.43），CINN 到 ~step400 被 `assert loss<=1` 中断（非 NaN）。
  - PP-YOLOE_plus-S：dynamic 跑满（4.79→1.44），**CINN 训练 NaN 发散**（检测 BCE loss 在 CINN+prim 下不稳）。
- PaddleX A100（2026-11-23，`gpu:0`，新版脚本 + 训练可复现补丁，产物 `results/a100/`）：
  - 推理：5 个模型同进程重复运行输出逐位一致，输入 sha256 与 MetaX 一致；dynamic/CINN 均 PASS（max_abs_diff ≤ 3.6e-7），耗时基本持平。
  - 训练（dynamic / CINN，最后 50 点均值；step 中位耗时）：StarNet-S4 2.987 / 2.664（-10.8%），227 / 150 ms；
    MobileNetV4 4.632 / 4.641，181 / 158 ms；OCR rec 10.72 / 10.52，100 / 65 ms；OCR det 1.478 / 1.443，186 / 149 ms；
    PP-YOLOE 1.606 / 1.621，102 / 78 ms。5 个模型的 CINN 训练均跑满、无 NaN/断言；首步 CINN 编译 4.9–9.5 min。
- PaddleScience 3 用例、PaddleFormers / PaddleClas 各目录：结果见各自文档。

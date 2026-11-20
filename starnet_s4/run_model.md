# StarNet-S4 CINN 复现（PaddleX 官方推理模型）

单卡、FP32、关闭 TF32。权重为 PaddleX 3.7 官方推理模型 `StarNet-S4`。PaddleX 对图像分类只发布静态推理图
（`inference.json` + `inference.pdiparams`），没有动态图训练代码，因此本目录验证的是**同一张 PIR 静态图**
在 `paddle.inference` 下的两种执行方式：

- dynamic：默认 PIR 执行器（`enable_new_ir` + `enable_new_executor`，不开 CINN）；
- cinn：同一张图加 `config.enable_cinn()`。

两种模式使用完全相同的权重文件和输入张量，只比较图执行结果与耗时。该配置与 PaddleX `PaddleStaticRunner`
中 `enable_cinn` 的用法一致，只是绕过了 predictor 的预处理/后处理，以便只测图执行。

## 环境

- 环境入口：`source /data/env.sh`（metax conda，`metax_gpu:0`）。
- Paddle：`3.5.0.dev20260516`，`paddle.is_compiled_with_cinn() == True`。cinn 模式入口会检查这一点，未编入 CINN 直接报错。
- PaddleX 源码：`/data/PaddleX-release-3.7`（`0.1.0.dev4387+gffb64904d`，即 release/3.7 `ffb64904`）。

## 数据与权重

权重不上传。默认从 `$PADDLEX_MODELS_DIR/StarNet-S4` 读取，未设置时为
`$HOME/.paddlex/official_models/StarNet-S4`；也可用 `CHECKPOINT_DIR` 指定。目录内需有
`inference.json`、`inference.pdiparams`、`inference.yml`，大小与 SHA256 见 `asset_manifest.json`，入口自动校验。

输入为固定种子的合成张量（`numpy.random.default_rng(20261119)`，`x: [1,3,224,224] float32`），
用于图级数值/性能对照，不是精度评测。真实预处理见 `inference.yml`（resize_short=256 → crop 224、ImageNet
mean/std），本目录不计入计时。

## 运行

```bash
source /data/env.sh
cd /data/Baidu_HIT_CiNN/starnet_s4
./run_model.sh        # dynamic
./run_model_cinn.sh   # cinn
# 可选：RESULTS_DIR=results/run2 ITERS=50 WARMUP=5 GPU_ID=0 MODE=... ./infer.sh
```

每个模式输出到 `results/<mode>-infer/`：`inference.json`（环境、shape、耗时）、`outputs.npz`（全部输出张量）、
运行日志。第二个模式跑完后会自动生成 `results/comparison.json`。已存在的结果目录不会被覆盖。

## 测量与验证

- warmup 5 次、计时 50 次，每次 run 前后 `paddle.device.synchronize()`；`steady_batch_ms` 只含图执行，
  不含预处理/H2D。`first_call_ms` 含 PIR 优化与 CINN 编译，不用于比较。
- CINN 生效判定：日志必须出现 `Compiling subgraph with CINN backend`，否则入口以退出码 4 失败。
- 数值比对：两模式 `outputs.npz` 逐元素最大绝对差 ≤ 2e-3（`compare.py`）。

2026-11-20 单次短测（metax_gpu:0）：dynamic 16.823 ms，cinn 16.675 ms（CINN 快 0.9%），
`max_abs_diff = 4.75e-5`，比对 PASS，比对 PASS。仅作入口/数值验证，不代表稳定性能结论。

## 跨平台（A100 / MetaX）复现

代码不绑定 Metax：`infer_impl.py` 在 CUDA 构建上自动用 `gpu:0`，否则用首个已加载的自定义设备
（`metax_gpu:0`），也可用 `PADDLE_DEVICE=gpu:0|metax_gpu:0|cpu` 强制。输入按固定种子生成，同时写入
`results/<mode>-infer/inputs.npz`；换平台时用 `INPUTS_NPZ=/path/to/inputs.npz` 复用同一批输入，
`inference.json` 里的 `inputs_sha256` 可确认两边输入完全一致。跨平台精度比对：

```bash
# MetaX 侧跑完 dynamic 后，把 results/dynamic-infer/ 整个目录拷到 A100
# A100 侧：
./run_model.sh
python compare.py --dynamic /path/to/metax/dynamic-infer \
                  --cinn results/dynamic-infer --report results/a100-vs-metax.json
```

跨平台差异按 `model_spec.json` 的 `atol` 判定（不同平台浮点实现不同），同平台 dynamic/CINN 差异通常再小 1~2 个数量级。

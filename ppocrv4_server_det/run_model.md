# PP-OCRv4_server_det CINN 复现（PaddleX 官方推理模型）

单卡、FP32、关闭 TF32。权重为 PaddleX 3.7 官方推理模型 `PP-OCRv4_server_det`。PaddleX 只发布静态推理图，
没有动态图训练代码，因此本目录验证的是**同一张 PIR 静态图**在 `paddle.inference` 下的两种执行方式：

- dynamic：默认 PIR 执行器（`enable_new_ir` + `enable_new_executor`，不开 CINN）；
- cinn：同一张图加 `config.enable_cinn()`。

## 环境

- 环境入口：`source /data/env.sh`（metax conda，`metax_gpu:0`）。
- Paddle：`3.5.0.dev20260516`，`paddle.is_compiled_with_cinn() == True`。
- PaddleX 源码：`/data/PaddleX-release-3.7`（`0.1.0.dev4387+gffb64904d`，release/3.7 `ffb64904`）。

## 数据与权重

权重不上传。默认从 `$PADDLEX_MODELS_DIR/PP-OCRv4_server_det` 读取（缺省
`$HOME/.paddlex/official_models/PP-OCRv4_server_det`），可用 `CHECKPOINT_DIR` 覆盖。大小与 SHA256 见
`asset_manifest.json`，入口自动校验。

输入为固定种子合成张量 `x: [1,3,960,960] float32`（960 对应 `DetResizeForTest resize_long=960`，且 32 的倍数；
该图 H/W 可按 32 的倍数变化）。输入用于图级数值/性能对照，不是 OCR 精度评测。

## 运行

```bash
source /data/env.sh
cd /data/Baidu_HIT_CiNN/ppocrv4_server_det
./run_model.sh        # dynamic
./run_model_cinn.sh   # cinn
# 可选：RESULTS_DIR=results/run2 ITERS=50 WARMUP=5 GPU_ID=0
```

产物为 `results/<mode>-infer/{inference.json,outputs.npz,*.log}`，第二个模式跑完自动生成
`results/comparison.json`；不覆盖已有结果目录。

## 测量与验证

- warmup 5、计时 50 次，run 前后 `paddle.device.synchronize()`，`steady_batch_ms` 只含图执行。
- CINN 生效判定：日志必须出现 `Compiling subgraph with CINN backend`，否则退出码 4。
- 数值比对：输出概率图 `[1,1,960,960]` 逐元素最大绝对差 ≤ 2e-3。

2026-11-20 单次短测（metax_gpu:0）：dynamic 53.199 ms，cinn 49.772 ms（CINN 快 6.9%），
`max_abs_diff = 3.86e-5`，比对 PASS，比对 PASS。仅作入口/数值验证，不代表稳定性能结论。

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

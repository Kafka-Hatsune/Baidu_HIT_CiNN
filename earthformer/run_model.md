# EarthFormer CuboidTransformer CINN 复现（PaddleScience case）

单卡、FP32、关闭 TF32。模型与合成批次复用 `$PADDLE_PSCI_DIR/test/cinn/cases/earthformer.py`（上游 CINN 测试套件
用例），本目录只负责两种执行方式、固定权重/输入和结果记录：

- dynamic：动态图前向（`no_grad`）；
- cinn：`paddle.jit.to_static(net, input_spec=..., full_graph=True, backend="CINN")`。

用例是 ENSO 预训练配置里的 `CuboidTransformer`（`examples/earthformer/conf/earthformer_enso_pretrain.yaml`
的 `MODEL` 段，保留 `checkpoint_level=2` 重算），批次为合成的
`sst_data [16,12,24,48,1]` 与 `sst_target [16,14,24,48,1]`，输出 2 个张量。

## 环境

- `source /data/env.sh`（metax conda）。
- Paddle `3.5.0.dev20260516`，`is_compiled_with_cinn() == True`。
- `export PADDLE_PSCI_DIR=/path/to/PaddleScience`，checkout `paddlescience.commit`（`34819214`），入口校验 HEAD。
- 设备自动选择：CUDA → `gpu:0`，否则 `metax_gpu:0`；`PADDLE_DEVICE` 可强制。
- 用例在静态图下会清空 `compute_cuboid_*_attention_mask` 的 `lru_cache`（缓存的 Tensor 在 to_static 下会跨
  程序泄漏）；这是用例内的必要处理，不要移除。

## 权重与输入

不使用外部权重。dynamic 在 CPU 上以固定种子初始化模型和批次，写入
`results/dynamic-infer/{init.pdparams,inputs.npz}`，cinn 自动复用；`inference.json` 的
`weights_sha256` / `inputs_sha256` 用于跨平台确认（A100 侧用 `CHECKPOINT_DIR`、`INPUTS_NPZ` 指定）。

## 运行

```bash
source /data/env.sh
export PADDLE_PSCI_DIR=/data/PaddleScience
cd /data/Baidu_HIT_CiNN/earthformer
./run_model.sh        # dynamic
./run_model_cinn.sh   # cinn
# 可选：RESULTS_DIR=results/run2 ITERS=20 WARMUP=5 GPU_ID=0 PADDLE_DEVICE=gpu:0
```

默认 `iters=20`（该模型单步约百毫秒）。产物：`results/<mode>-infer/`、`results/comparison.json`；
不覆盖已有结果目录。

## 测量与验证

- warmup 5、计时 20 次，run 前后 `paddle.device.synchronize()`。
- CINN 生效判定：日志必须出现 `Compiling subgraph with CINN backend`，否则退出码 4。
- 数值比对：2 个输出张量逐个比较，最大绝对差 ≤ 2e-3（输出量级约 15，实测相对误差 ~6e-5）。

2026-11-20 单次短测（metax_gpu:0）：dynamic 114.313 ms，cinn 46.382 ms（**CINN 快 146.5%**，约 2.5×），
`max_abs_diff = 9.03e-4`，比对 PASS；`weights_sha256` / `inputs_sha256` 一致。

训练：上游 harness 支持 `python test/cinn/run_cinn_test.py --cases earthformer --stages train`；
本目录只覆盖推理前向。

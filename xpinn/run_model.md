# XPINN 2D Poisson CINN 复现（PaddleScience case）

单卡、FP32、关闭 TF32。模型与合成批次复用 `$PADDLE_PSCI_DIR/test/cinn/cases/xpinn.py`（上游 CINN 测试套件用例），
本目录只负责两种执行方式、固定权重/输入和结果记录：

- dynamic：动态图 `model(inputs)`（`no_grad`）；
- cinn：`paddle.jit.to_static(net, input_spec=..., full_graph=True, backend="CINN")`。

用例是 3 个子域的 MLP（`[2,30,30,1]`、`[2,20,20,20,20,1]`、`[2,25,25,25,1]`），输入为 6 组合成点
（3 个残差点集 + 2 个界面 + 边界），输出 18 个张量（含 12 个原样透传的坐标张量）。

## 环境

- `source /data/env.sh`（metax conda）。
- Paddle `3.5.0.dev20260516`，`is_compiled_with_cinn() == True`。
- `export PADDLE_PSCI_DIR=/path/to/PaddleScience`，checkout `paddlescience.commit`（`34819214`），入口校验 HEAD。
- 设备自动选择：CUDA → `gpu:0`，否则 `metax_gpu:0`；`PADDLE_DEVICE` 可强制。
- 注意：用例模块会导入 `examples/xpinn/model.py` 与 `examples/xpinn/xpinn.py`，后者像示例一样调用
  `set_prim_eager_enabled(True)`；这是用例的一部分，两端一致。

## 权重与输入

不使用外部权重。dynamic 在 CPU 上以固定种子初始化模型和批次并写入
`results/dynamic-infer/{init.pdparams,inputs.npz}`，cinn 自动复用；`inference.json` 的
`weights_sha256` / `inputs_sha256` 用于跨平台确认（A100 侧用 `CHECKPOINT_DIR`、`INPUTS_NPZ` 指定这两份文件）。

## 运行

```bash
source /data/env.sh
export PADDLE_PSCI_DIR=/data/PaddleScience
cd /data/Baidu_HIT_CiNN/xpinn
./run_model.sh        # dynamic
./run_model_cinn.sh   # cinn
# 可选：RESULTS_DIR=results/run2 ITERS=50 WARMUP=5 GPU_ID=0 PADDLE_DEVICE=gpu:0
```

产物：`results/<mode>-infer/`、`results/comparison.json`；不覆盖已有结果目录。

## 测量与验证

- warmup 5、计时 50 次，run 前后 `paddle.device.synchronize()`。
- CINN 生效判定：日志必须出现 `Compiling subgraph with CINN backend`，否则退出码 4。
- 数值比对：18 个输出张量逐个比较，最大绝对差 ≤ 2e-3。

2026-11-20 单次短测（metax_gpu:0）：dynamic 1.789 ms，cinn 3.640 ms（CINN 慢 50.8%），
`max_abs_diff = 0.0`，比对 PASS；`weights_sha256` / `inputs_sha256` 一致。小模型 + 18 个输出（含透传张量）时
CINN 的调度开销占比高，变慢是预期现象，仅作入口/数值验证。

训练（`train.sh`，复用上游 case 的 `compute_loss`，默认 `WARMUP=5`、`ITERS=500`）：

```bash
./train.sh              # dynamic
MODE=cinn ./train.sh    # CINN；自动复用 dynamic-train 的 init.pdparams 与 batch.npz
```

产物 `results/{dynamic,cinn}-train/`，比对写入 `results/comparison-train.json`（400-500 窗口、tol 1e-2、
最多 10 个离群步）。跨平台用 `CHECKPOINT_DIR` + `BATCH_NPZ` 复用权重/批次。

2026-11-20 实测（metax_gpu:0，505 步）：dynamic 120.55 ms/step，CINN 95.86 ms/step（1.26×）；
首步 loss 差 9.0e-8，400-500 窗口最大差 2.74e-5、0 个离群步 → **PASS**。

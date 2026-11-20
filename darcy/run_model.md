# darcy 2D Darcy flow CINN 复现（PaddleScience case）

单卡、FP32、关闭 TF32。模型与合成批次直接复用 `$PADDLE_PSCI_DIR/test/cinn/cases/darcy.py`（PaddleScience 上游
CINN 测试套件的用例），本目录只负责两种执行方式、固定权重/输入和结果记录：

- dynamic：动态图 `model(inputs)`（`no_grad`）；
- cinn：`paddle.jit.to_static(net, input_spec=..., full_graph=True, backend="CINN")`。

两种模式加载同一份 `init.pdparams` 与 `inputs.npz`。用例为 5×20、`stan` 激活的 MLP，损失是 Poisson 方程
残差 + 边界条件（训练部分见文末）。

## 环境

- `source /data/env.sh`（metax conda）。
- Paddle `3.5.0.dev20260516`，`is_compiled_with_cinn() == True`；cinn 模式入口会检查。
- `export PADDLE_PSCI_DIR=/path/to/PaddleScience`，并 checkout `paddlescience.commit`（当前
  `34819214`）；入口会校验 HEAD，不一致直接退出。
- 设备自动选择：CUDA 构建 → `gpu:0`；否则首个自定义设备 → `metax_gpu:0`；`PADDLE_DEVICE` 可强制。

## 权重与输入

不使用外部权重。dynamic 运行在 **CPU** 上以固定种子（`model_spec.json` 的 `seed`）初始化模型与合成批次，
写入 `results/dynamic-infer/{init.pdparams,inputs.npz}`；cinn 运行自动复用这两份文件。CPU 随机流与设备无关，
所以 A100 与 MetaX 可以从完全相同的参数/输入出发；`inference.json` 的 `weights_sha256`、`inputs_sha256`
用于确认。跨平台时把这两个文件拷过去，用 `CHECKPOINT_DIR`、`INPUTS_NPZ` 指定即可。

## 运行

```bash
source /data/env.sh
export PADDLE_PSCI_DIR=/data/PaddleScience
cd /data/Baidu_HIT_CiNN/darcy
./run_model.sh        # dynamic
./run_model_cinn.sh   # cinn
# 可选：RESULTS_DIR=results/run2 ITERS=50 WARMUP=5 GPU_ID=0 PADDLE_DEVICE=gpu:0
```

产物：`results/<mode>-infer/{inference.json,outputs.npz,init.pdparams,inputs.npz}`、日志，以及
`results/comparison.json`（第二个模式跑完后自动生成）。已存在的结果目录不会被覆盖。

## 测量与验证

- warmup 5、计时 50 次，run 前后 `paddle.device.synchronize()`；`steady_batch_ms` 只含前向执行。
- CINN 生效判定：日志必须出现 `Compiling subgraph with CINN backend`，否则退出码 4。
- 数值比对：`outputs.npz` 逐张量最大绝对差 ≤ 2e-3（`model_spec.json` 的 `atol`）。

2026-11-20 单次短测（metax_gpu:0）：dynamic 0.981 ms，cinn 1.061 ms（CINN 慢 7.5%），
`max_abs_diff = 0.0`，比对 PASS；两模式 `weights_sha256` / `inputs_sha256` 一致。仅作入口/数值验证。

训练（`train.sh`，复用上游 case 的 `compute_loss`，默认 `WARMUP=5`、`ITERS=500`）：

```bash
./train.sh              # dynamic
MODE=cinn ./train.sh    # CINN；自动复用 dynamic-train 的 init.pdparams 与 batch.npz
```

产物 `results/{dynamic,cinn}-train/`（`train.json`、`losses.npz`、`init.pdparams`、`batch.npz`、`final.pdparams`），
`results/comparison-train.json` 按 400-500 窗口、tol 1e-2、最多 10 个离群步判定。跨平台用
`CHECKPOINT_DIR` + `BATCH_NPZ` 复用另一端的权重/批次，`train.json` 的 `init_weights_sha256`、`batch_sha256` 用于核对。

2026-11-20 实测（metax_gpu:0，505 步）：dynamic 38.96 ms/step，CINN 18.51 ms/step（2.1×）；
首步 loss 差 6.6e-8（同权重同批次），400-500 窗口最大差 1.24e-2、均值 2.2e-3、1 个离群步 → **PASS**。
上游 harness 在同一机器默认口径下曾 ACC_FAIL（5.6e-2、37 个离群步），两者初始权重/批次生成方式不同
（上游在设备上生成），只适合各自比较 dynamic↔CINN，不能横向对比数值。

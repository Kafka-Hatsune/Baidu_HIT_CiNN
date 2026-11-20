本目录不使用外部预训练权重。`dynamic` 运行在 CPU 上以固定种子（`model_spec.json` 的 `seed`）
初始化模型，并把结果写到 `results/dynamic-infer/init.pdparams` 与 `inputs.npz`；`cinn` 运行自动复用
这两份文件（也可用 `CHECKPOINT_DIR` / `INPUTS_NPZ` 指定别的平台导出的同一份权重与输入），
保证两端加载完全相同的参数与批次。权重和输入的 SHA256 记录在各自的 `inference.json` 中。

训练入口（`train.sh`）同样不使用外部权重：dynamic 运行生成 `results/dynamic-train/{init.pdparams,batch.npz,batch.json}`，
cinn 运行自动复用；也可用 `CHECKPOINT_DIR`（init.pdparams）与 `BATCH_NPZ`（batch.npz，需同目录有同名 `.json`）
复用另一平台的权重与批次，保证两端训练轨迹从同一点出发。

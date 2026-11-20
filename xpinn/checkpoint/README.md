本目录不使用外部预训练权重。`dynamic` 运行在 CPU 上以固定种子（`model_spec.json` 的 `seed`）
初始化模型，并把结果写到 `results/dynamic-infer/init.pdparams` 与 `inputs.npz`；`cinn` 运行自动复用
这两份文件（也可用 `CHECKPOINT_DIR` / `INPUTS_NPZ` 指定别的平台导出的同一份权重与输入），
保证两端加载完全相同的参数与批次。权重和输入的 SHA256 记录在各自的 `inference.json` 中。

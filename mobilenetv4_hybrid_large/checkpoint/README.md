官方 PaddleX 推理权重暂不上传。外部目录提供 `inference.json`、`inference.pdiparams`、
`inference.yml`；默认从 `$PADDLEX_MODELS_DIR/MobileNetV4_hybrid_large`（缺省
`$HOME/.paddlex/official_models/MobileNetV4_hybrid_large`）读取，也可用 `CHECKPOINT_DIR` 指定。
大小与 SHA256 见 `../asset_manifest.json`，入口自动校验。

下载方式（联网机器）：`PX=/path/to/PaddleX python -c "from paddlex.inference.utils.official_models import official_models as o; print(o.get_model_path('MobileNetV4_hybrid_large'))"`，

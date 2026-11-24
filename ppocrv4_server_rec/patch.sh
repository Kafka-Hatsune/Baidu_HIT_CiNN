#!/usr/bin/env bash
# Apply / check / revert the local source patches this model's training needs
# in the PaddleX-installed repos (PaddleOCR / PaddleDetection). Idempotent:
# train.sh runs `patch.sh apply` before every training. See README.md「源码补丁」.
#
#   ./patch.sh [apply|check|revert]      # default: apply
#
# The seed patches only take effect when CINN_TRAIN_SEED is set (train.sh sets
# it unless DETERMINISTIC=0), so patched repos behave like upstream otherwise.
set -euo pipefail

PATCHES="ocr_metax_device ocr_dataset_seed"
ACTION=${1:-apply}
PADDLEX=${PADDLEX:-/data/PaddleX-release-3.7}
REPOS="$PADDLEX/paddlex/repo_manager/repos"
case "$ACTION" in apply|check|revert) ;; *) echo "usage: $0 [apply|check|revert]" >&2; exit 2 ;; esac
[[ -d "$REPOS" ]] || { echo "PaddleX repos not found: $REPOS (set PADDLEX)" >&2; exit 2; }

exec "${PYTHON_BIN:-python}" - "$ACTION" "$REPOS" $PATCHES <<'EOF'
import os
import sys

action, repos, names = sys.argv[1], sys.argv[2], sys.argv[3:]

# name -> (file relative to repos/, [(upstream text, patched text), ...])
PATCHES = {
    # PaddleOCR builds the metax device string as "metax:N"; paddle expects "metax_gpu:N".
    "ocr_metax_device": ("PaddleOCR/tools/program.py", [(
        'device = "metax:{0}".format(dist.ParallelEnv().dev_id)',
        'device = "metax_gpu:{0}".format(dist.ParallelEnv().dev_id)',
    )]),
    # SimpleDataSet is built with seed=None (the arg doubles as epoch number), and
    # random.seed(None) reseeds from OS entropy, so sample order changes every run.
    "ocr_dataset_seed": ("PaddleOCR/ppocr/data/simple_dataset.py", [
        ("import concurrent.futures\n",
         "import concurrent.futures\n"
         "\n"
         "\n"
         "def _cinn_seed(seed):\n"
         "    # Baidu_HIT_CiNN patch: random.seed(None) reseeds from OS entropy; use\n"
         "    # CINN_TRAIN_SEED instead so the sample order is reproducible.\n"
         "    if seed is None and os.environ.get(\"CINN_TRAIN_SEED\"):\n"
         "        return int(os.environ[\"CINN_TRAIN_SEED\"])\n"
         "    return seed\n"),
        ("                random.seed(seed)\n                sampled.extend",
         "                random.seed(_cinn_seed(seed))\n                sampled.extend"),
        ("            random.seed(seed)\n            random.shuffle(sampled)",
         "            random.seed(_cinn_seed(seed))\n            random.shuffle(sampled)"),
        ("                    random.seed(self.seed)\n                    lines = random.sample",
         "                    random.seed(_cinn_seed(self.seed))\n                    lines = random.sample"),
        ("        random.seed(self.seed)\n        random.shuffle(self.data_lines)",
         "        random.seed(_cinn_seed(self.seed))\n        random.shuffle(self.data_lines)"),
    ]),
    # albumentations.Compose without seed draws from OS entropy and ignores
    # np.random.seed; seed it, offset per DataLoader worker.
    "ocr_iaa_seed": ("PaddleOCR/ppocr/data/imaug/iaa_augment.py", [
        ("                return A.Compose(\n                    sequence,\n",
         "                return A.Compose(\n                    sequence,\n"
         "                    # Baidu_HIT_CiNN patch: reproducible augmentation.\n"
         "                    seed=int(os.environ[\"CINN_TRAIN_SEED\"])\n"
         "                    if os.environ.get(\"CINN_TRAIN_SEED\")\n"
         "                    else None,\n"),
        ("    def __call__(self, data):\n        image = data[\"image\"]\n\n        if self.augmenter:\n",
         "    def __call__(self, data):\n        image = data[\"image\"]\n\n"
         "        # Baidu_HIT_CiNN patch: forked DataLoader workers copy the same seeded\n"
         "        # state; offset it by worker id so workers draw independent streams.\n"
         "        if (\n"
         "            self.augmenter\n"
         "            and self.augmenter.seed is not None\n"
         "            and not getattr(self, \"_worker_seeded\", False)\n"
         "        ):\n"
         "            self._worker_seeded = True\n"
         "            from paddle.io import get_worker_info\n"
         "\n"
         "            info = get_worker_info()\n"
         "            if info is not None:\n"
         "                self.augmenter.set_random_seed(self.augmenter.seed + info.id)\n"
         "\n"
         "        if self.augmenter:\n"),
    ]),
    # PaddleX never passes --enable_ce, so PaddleDetection seeds nothing: the new
    # class head initializes randomly and augmentation differs every run.
    "ppdet_seed": ("PaddleDetection/tools/train.py", [(
        "    if FLAGS.enable_ce:\n        set_random_seed(0)\n",
        "    if FLAGS.enable_ce:\n        set_random_seed(0)\n"
        "    # Baidu_HIT_CiNN patch: PaddleX never passes --enable_ce; seed from env.\n"
        "    elif os.environ.get(\"CINN_TRAIN_SEED\"):\n"
        "        set_random_seed(int(os.environ[\"CINN_TRAIN_SEED\"]))\n",
    )]),
}

failed = False
for name in names:
    rel, pairs = PATCHES[name]
    path = os.path.join(repos, rel)
    text = open(path).read()
    new_text, state = text, set()
    for old, new in pairs:
        if new in text:
            state.add("patched")
            if action == "revert":
                new_text = new_text.replace(new, old, 1)
        elif text.count(old) == 1:
            state.add("upstream")
            if action == "apply":
                new_text = new_text.replace(old, new, 1)
        else:
            state.add("unknown")
    if "unknown" in state:
        print(f"{name}: {rel} does not match the expected upstream code, patch by hand", file=sys.stderr)
        failed = True
        continue
    if new_text != text:
        with open(path, "w") as f:
            f.write(new_text)
    status = {"apply": "applied", "revert": "reverted"}.get(action) if new_text != text else (
        "patched" if state == {"patched"} else "not patched" if state == {"upstream"} else "partially patched")
    print(f"{name}: {status} ({rel})")
    if action == "check" and state != {"patched"}:
        failed = True
sys.exit(1 if failed else 0)
EOF

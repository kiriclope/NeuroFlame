python src/train/dual/launch_dual.py \
  --task dpa \
  --n-runs 1 \
  --n-batch 128 \
  --start-seed 1 \
  --gpus 0 1 \
  --max-procs-per-gpu 1 \
  --out-root src/train/runs/dual \
  --log-root src/train/logs/dual

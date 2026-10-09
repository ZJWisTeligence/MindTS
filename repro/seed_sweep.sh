#!/bin/bash
# 多种子排查: 用官方超参数, 只换种子, 看 MDT / EWJ 的结果波动有多大
# 用法: bash repro/seed_sweep.sh "MDT EWJ" "2021 1 2 3 4"
# 默认用 bf16 加速 (已验证 bf16 与 fp32 在 MDT 上相差 <0.4 点); 需要 fp32 时设 MINDTS_BF16=0
cd "$(dirname "$0")/.." || exit 1
export PATH=/opt/conda/envs/mindts/bin:$PATH
export MINDTS_BF16=${MINDTS_BF16:-1}
datasets=${1:-"MDT EWJ"}
seeds=${2:-"2021 1 2 3 4"}
mkdir -p repro/logs/seed_sweep

for name in $datasets; do
  script=$(ls scripts/*/detect_label/${name}_script/MindTS.sh)
  for seed in $seeds; do
    echo "[$(date '+%F %T')] start $name seed=$seed"
    start=$(date +%s)
    # 只替换保存路径, 让每个种子的结果单独存放
    cmd=$(sed "s#--save-path \"label/$name\"#--save-path \"seed_sweep/$name/s$seed\"#" "$script")
    MINDTS_SEED=$seed sh -c "$cmd" > "repro/logs/seed_sweep/${name}_s$seed.log" 2>&1
    rc=$?
    echo "[$(date '+%F %T')] done  $name seed=$seed exit=$rc $(( $(date +%s) - start ))s"
  done
done
echo ALL_DONE

#!/bin/bash
# 修复版 有文本 vs 无外生文本, 每个数据集 5 个种子, 最多 5 个并行
# 用法: bash repro/run_fix.sh <输出前缀, 如 fix> "<数据集列表, 如 MDT EWJ>"
#   结果存到 result/<前缀>/ 和 result/<前缀>_notext/; 修复开关由调用方用环境变量传入 (MINDTS_FIX_*)
prefix=${1:-fix}; datasets=${2:-"MDT EWJ"}
cd "$(dirname "$0")/.." || exit 1
mkdir -p "repro/logs/$prefix" "repro/logs/${prefix}_notext"
export PATH=/opt/conda/envs/mindts/bin:$PATH MINDTS_BF16=1
jobs_list=""; for d in $datasets; do for s in 2021 1 2 3 4; do jobs_list="$jobs_list $d:$s"; done; done
for j in $jobs_list; do
  n=${j%%:*}; s=${j##*:}; sc=$(ls scripts/*/detect_label/${n}_script/MindTS.sh)
  for v in "$prefix" "${prefix}_notext"; do
    while [ "$(jobs -rp | wc -l)" -ge 5 ]; do sleep 2; done
    (
      st=$(date +%s)
      if [ "$v" = "$prefix" ]; then
        cmd=$(sed "s#--save-path \"label/$n\"#--save-path \"$v/$n/s$s\"#" "$sc")
        MINDTS_SEED=$s sh -c "$cmd" > repro/logs/$v/${n}_s$s.log 2>&1
      else
        MINDTS_SEED=$s python repro/ablate_notext.py "$sc" "$v/$n/s$s" > repro/logs/$v/${n}_s$s.log 2>&1
      fi
      rc=$?; echo "[$(date '+%T')] done $v $n s=$s exit=$rc $(( $(date +%s) - st ))s"
    ) &
    sleep 15
  done
done
wait; echo ALL_DONE

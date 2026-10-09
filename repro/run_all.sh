#!/bin/bash
# 依次复现论文 Table 1 中 6 个数据集的 MindTS 结果, 日志写到 repro/logs/
cd "$(dirname "$0")/.." || exit 1
export PATH=/opt/conda/envs/mindts/bin:$PATH
mkdir -p repro/logs

for s in \
  univariate_detection/detect_label/MDT_script \
  univariate_detection/detect_label/KR_script \
  univariate_detection/detect_label/EWJ_script \
  univariate_detection/detect_label/Environment_script \
  multivariate_detection/detect_label/Energy_script \
  multivariate_detection/detect_label/Weather_script; do
  name=$(basename "$s" _script)
  echo "[$(date '+%F %T')] start $name"
  start=$(date +%s)
  sh "./scripts/$s/MindTS.sh" > "repro/logs/$name.log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] done  $name exit=$rc $(( $(date +%s) - start ))s"
done
echo ALL_DONE

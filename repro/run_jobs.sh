#!/bin/bash
# 并行跑多个 (数据集, 种子) 任务, 充分利用显存
# 用法: bash repro/run_jobs.sh <并发数> <输出子目录> "MDT:1 EWJ:2021 ..."
#   输出子目录: 结果存到 result/<子目录>/<数据集>/s<种子>, 日志存到 repro/logs/<子目录>/
#   额外环境变量: MINDTS_BF16 (默认 1), EXTRA_PARAMS (追加/覆盖超参数, 如 '"num_epochs": 3')
cd "$(dirname "$0")/.." || exit 1
export PATH=/opt/conda/envs/mindts/bin:$PATH
export MINDTS_BF16=${MINDTS_BF16:-1}
jobs_max=$1; out=$2; jobs_list=$3
mkdir -p "repro/logs/$out"

run_one() {
  name=${1%%:*}; seed=${1##*:}
  script=$(ls scripts/*/detect_label/${name}_script/MindTS.sh)
  cmd=$(sed "s#--save-path \"label/$name\"#--save-path \"$out/$name/s$seed\"#" "$script")
  if [ -n "$EXTRA_PARAMS" ]; then
    # 在超参数 JSON 末尾的 } 前追加, JSON 中后出现的同名键覆盖前面的
    cmd=$(echo "$cmd" | sed "s#\"parallel_strategy\": \"DP\"}#\"parallel_strategy\": \"DP\", $EXTRA_PARAMS}#")
  fi
  echo "[$(date '+%F %T')] start $name seed=$seed"
  start=$(date +%s)
  MINDTS_SEED=$seed sh -c "$cmd" > "repro/logs/$out/${name}_s$seed.log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] done  $name seed=$seed exit=$rc $(( $(date +%s) - start ))s"
}

for j in $jobs_list; do
  while [ "$(jobs -rp | wc -l)" -ge "$jobs_max" ]; do sleep 2; done
  run_one "$j" &
  sleep 20  # 错开模型加载, 避免同时读 3.5GB 权重
done
wait
echo ALL_DONE

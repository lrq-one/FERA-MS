#!/usr/bin/env bash

cd /home/lwh/projects/lrq2/fragnnet-main/ms2spectra_v1_r119 || exit 1

watch -n 5 '
OUT="runs/experiments/molecular_retrieval/pubchem_legacy_full/ours_r184b_experiment5_20260724"
LOG="$OUT/run.log"

echo "================ EXPERIMENT 5 ================"
date

echo
echo "=== 后台进程 ==="
pgrep -af "[r]un_experiment5_ours_supervisor.sh" || true
pgrep -af "[p]ython -u test/run_experiment5_ours.py" || true

echo
echo "=== GPU ==="
nvidia-smi \
  --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv,noheader 2>/dev/null || true

echo
echo "=== 最新断点/进度 ==="
grep -E \
  "\[resume\]|\[checkpoint\]|\[progress\]|\[predictions complete\]|\[skip complete\]|EXPERIMENT 5 COMPLETE|\[all complete\]|Traceback|RuntimeError|CUDA out of memory|\[supervisor\]" \
  "$LOG" 2>/dev/null | tail -n 35

echo
echo "=== 已完成组合 ==="
find "$OUT" -path "*/seed_*/_SUCCESS.json" \
  -printf "%TY-%Tm-%Td %TH:%TM  %h\n" \
  2>/dev/null | sort

echo
echo "=== 每组指标文件 ==="
find "$OUT" -path "*/seed_*/retrieval_summary.csv" \
  -printf "%TY-%Tm-%Td %TH:%TM  %10s  %p\n" \
  2>/dev/null | sort

echo
echo "=== 当前汇总表 ==="
if [ -s "$OUT/experiment5_ours_aggregate.csv" ]; then
    python -c "
import pandas as pd
p=\"${OUT}/experiment5_ours_aggregate.csv\"
d=pd.read_csv(p)
cols=[
    c for c in [
        \"split\",\"cohort\",\"method\",\"unit\",\"seed_count\",
        \"top1_mean\",\"top5_mean\",\"top10_mean\",\"mrr_mean\",
        \"median_rank_mean\"
    ] if c in d.columns
]
print(d[cols].to_string(index=False))
" 2>/dev/null || true
else
    echo "尚无完整组合结果"
fi
'

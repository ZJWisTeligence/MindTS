"""汇总多种子排查结果: 每个种子一行, 最后给出均值 / 标准差 / 最大值, 并与论文对比."""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import PAPER, load_latest  # noqa: E402

# 用法: python repro/summarize_seeds.py [结果子目录, 默认 seed_sweep]
SWEEP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "result",
                     sys.argv[1] if len(sys.argv) > 1 else "seed_sweep")

for name in sorted(os.listdir(SWEEP)) if os.path.isdir(SWEEP) else []:
    rows = []
    for d in sorted(glob.glob(os.path.join(SWEEP, name, "s*"))):
        df = load_latest(os.path.basename(d), root=os.path.join(SWEEP, name))
        if df is None:
            continue
        rows.append([os.path.basename(d)[1:], df["affiliation_f"].max() * 100,
                     df["VUS_PR"].iloc[0] * 100, df["VUS_ROC"].iloc[0] * 100])
    if not rows:
        continue
    t = pd.DataFrame(rows, columns=["seed", "Aff-F", "V-PR", "V-ROC"])
    m = t[["Aff-F", "V-PR", "V-ROC"]]
    stats = pd.DataFrame([["mean", *m.mean()], ["std", *m.std(ddof=0)], ["max", *m.max()],
                          ["paper", *PAPER[name]]], columns=t.columns)
    print(f"== {name} ({len(t)} seeds)")
    print(pd.concat([t, stats]).to_string(index=False, float_format="%.2f"))
    print()

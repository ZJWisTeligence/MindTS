"""汇总每个数据集最新一次运行的结果, 与论文 Table 1 对比.

Aff-F 取所有 anomaly_ratio 中的最大值; V-PR / V-ROC 与阈值无关.
"""
import glob
import io
import os
import tarfile

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "result", "label")

# 论文 Table 1 中 MindTS 的结果 (Aff-F, V-PR, V-ROC)
PAPER = {
    "MDT": (89.19, 65.44, 83.02),
    "KR": (90.28, 53.15, 89.86),
    "EWJ": (83.89, 50.42, 84.12),
    "Environment": (85.29, 56.79, 93.78),
    "Energy": (74.37, 50.36, 74.44),
    "Weather": (82.66, 57.48, 82.64),
}


def load_latest(name, root=ROOT):
    files = sorted(glob.glob(os.path.join(root, name, "MindTS.*.csv.tar.gz")), key=os.path.getmtime)
    if not files:
        return None
    frames = []
    with tarfile.open(files[-1]) as tar:
        for m in tar.getmembers():
            if m.isfile():
                frames.append(pd.read_csv(io.BytesIO(tar.extractfile(m).read())))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    rows = []
    for name, (p_aff, p_vpr, p_vroc) in PAPER.items():
        df = load_latest(name)
        if df is None:
            rows.append([name, "未运行"] + [""] * 6)
            continue
        best = df.loc[df["affiliation_f"].idxmax()]
        aff, vpr, vroc = best["affiliation_f"] * 100, df["VUS_PR"].iloc[0] * 100, df["VUS_ROC"].iloc[0] * 100
        rows.append([name, int(best["typical_anomaly_ratio"]),
                     f"{aff:.2f} / {p_aff:.2f}", f"{aff - p_aff:+.2f}",
                     f"{vpr:.2f} / {p_vpr:.2f}", f"{vpr - p_vpr:+.2f}",
                     f"{vroc:.2f} / {p_vroc:.2f}", f"{vroc - p_vroc:+.2f}"])

    out = pd.DataFrame(rows, columns=["Dataset", "best_ratio", "Aff-F (ours/paper)", "ΔAff-F",
                                      "V-PR (ours/paper)", "ΔV-PR", "V-ROC (ours/paper)", "ΔV-ROC"])
    print(out.to_string(index=False))

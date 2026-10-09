"""[临时排查] 在真实运行流程中截获送进模型的外生文本, 统计有效 token 数, 打印样例后退出.
用法: python repro/probe_text.py <官方脚本路径>
不修改任何项目文件, 只在本进程内给 MINDTSModel.forward 打补丁.
"""
import os
import runpy
import shlex
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from ts_benchmark.baselines.MindTS.models import MindTS_model as M  # noqa: E402

stats = {"n": 0, "valid": [], "full": 0, "samples": []}
_orig = M.MINDTSModel.forward


def probed(self, x_enc_time, x_enc_input_ids, x_enc_attention_mask):
    mask = x_enc_attention_mask.long()
    valid = mask.sum(dim=1).cpu().numpy()
    stats["valid"].extend(valid.tolist())
    stats["full"] += int((valid >= mask.shape[1]).sum())
    if len(stats["samples"]) < 3:
        ids = x_enc_input_ids.long()[0][mask[0].bool()]
        stats["samples"].append(self.tokenizer.decode(ids[:200]))
    stats["n"] += 1
    if stats["n"] >= 20:
        v = np.array(stats["valid"])
        print("\n========== PROBE ==========")
        print(f"batches={stats['n']} samples={len(v)} max_len={mask.shape[1]}")
        print(f"valid tokens per sample: mean={v.mean():.0f} median={np.median(v):.0f} "
              f"min={v.min()} max={v.max()}  truncated(=1024)={stats['full']}/{len(v)}")
        for i, s in enumerate(stats["samples"]):
            print(f"--- sample {i} (first 200 tokens) ---\n{s[:600]}")
        sys.stdout.flush()  # os._exit 不会刷新缓冲区
        os._exit(0)
    return _orig(self, x_enc_time, x_enc_input_ids, x_enc_attention_mask)


M.MINDTSModel.forward = probed

# 从官方脚本里解析出 run_benchmark.py 的参数, 在本进程内执行
cmd = shlex.split(open(sys.argv[1]).read().strip())
sys.argv = cmd[1:]
runpy.run_path(cmd[1], run_name="__main__")

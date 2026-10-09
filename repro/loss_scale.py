"""[临时排查] 记录训练时三项损失 (重建 MSE / 对齐 clip_loss / 压缩 Bottleneck_loss) 的实际大小, 第 1 个 epoch 结束后打印并退出.
用法: MINDTS_SEED=2021 python repro/loss_scale.py <官方脚本路径>
不修改任何项目文件, 只在本进程内打补丁.
"""
import os
import runpy
import shlex
import sys

import numpy as np
import torch.nn as nn

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import importlib  # noqa: E402

# 包的 __init__ 导出了同名类, 用 import_module 才能拿到模块本身
mod = importlib.import_module("ts_benchmark.baselines.MindTS.MindTS")

rec = {"mse": [], "clip": [], "bottleneck": [], "shape": None}
_clip, _bott, _mse = mod.clip_loss, mod.Bottleneck_loss, nn.MSELoss.forward


def clip_rec(lt, lx):
    v = _clip(lt, lx)
    rec["clip"].append(v.item())
    rec["shape"] = tuple(lt.shape)
    return v


def bott_rec(m, r, lamda):
    v = _bott(m, r, lamda)
    rec["bottleneck"].append(v.item())
    return v


def mse_rec(self, a, b):
    v = _mse(self, a, b)
    if self.reduction == "mean":
        rec["mse"].append(v.item())
    return v


mod.clip_loss, mod.Bottleneck_loss, nn.MSELoss.forward = clip_rec, bott_rec, mse_rec


def stop(self, loader, criterion):
    # 第 1 个 epoch 训练结束, 进入验证前打印统计
    def s(k, a, b):
        x = np.array(rec[k][a:b])
        return f"{x.mean():.4f}"
    n = len(rec["clip"])
    print("\n========== LOSS SCALE (train, epoch 1) ==========")
    print(f"steps={n}  logits_per_time shape={rec['shape']}  (B*C, N, N), "
          f"随机猜测时每个样本的 CE = ln(N) = {np.log(rec['shape'][1]):.3f}")
    for k in ["mse", "clip", "bottleneck"]:
        print(f"{k:>10}: 前10步均值={s(k, 0, 10)}  后10步均值={s(k, n - 10, n)}")
    c, m = np.mean(rec["clip"][-10:]), np.mean(rec["mse"][-10:])
    print(f"后10步: clip/mse = {c / m:.1f} 倍;  clip 每个样本平均 = {c / rec['shape'][0]:.3f}")
    sys.stdout.flush()
    os._exit(0)


mod.MindTS.detect_multi_validate = stop

cmd = shlex.split(open(sys.argv[1]).read().strip())
args = cmd[1:]
args[args.index("--save-path") + 1] = "diag_tmp"
sys.argv = args
runpy.run_path(args[0], run_name="__main__")

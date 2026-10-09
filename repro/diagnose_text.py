"""[临时排查] 文本分支诊断: 正常训练完一个模型后, 在测试数据上测量文本路径的实际贡献, 打印后退出.
用法: MINDTS_SEED=2021 python repro/diagnose_text.py <官方脚本路径>
不修改任何项目文件, 只在本进程内打补丁. 结果不写入 result/.
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

from ts_benchmark.baselines.MindTS.MindTS import MindTS as W  # noqa: E402  (类本身)
from ts_benchmark.baselines.MindTS.models import MindTS_model as M  # noqa: E402

# 1) 记录 proj_text / proj_prompt / proj_patch 的初始权重, 训练后看改变了多少
_orig_init = M.MINDTSModel.__init__


def init_and_snapshot(self, configs):
    _orig_init(self, configs)
    self._w0 = {k: getattr(self, k).weight.detach().cpu().clone()
                for k in ["proj_text", "proj_prompt", "proj_patch", "prompt_proj_hidden", "text_proj_hidden"]
                if getattr(self, k) is not None}


M.MINDTSModel.__init__ = init_and_snapshot

N_BATCH = 8
_orig_label = W.detect_multi_label


def norm(t):
    return t.float().norm(dim=-1).mean().item()


def diagnose(self, test_data, test_text):
    res = _orig_label(self, test_data, test_text)  # 正常完成评估 (加载最优 checkpoint)
    m = self.model
    m.eval()
    dev = m.device
    out = {"psi": [], "text_attn": [], "prompt": [], "cross": [], "time_mask": [],
           "text_var_across_samples": [], "text_norm": []}
    hooks = []
    rec = {"on": False}

    def h_tb(mod, inp, kw, outp):
        if not rec["on"]:
            return
        prompt, text = inp[0], inp[1]
        kpm = kw.get("key_padding_mask")
        attn, _ = mod.attention(prompt, text, text, key_padding_mask=kpm)
        out["prompt"].append(norm(prompt)); out["text_attn"].append(norm(attn))
        if kpm is None:
            t = text.mean(1)
        else:
            v = (~kpm).unsqueeze(-1).float()
            t = (text * v).sum(1) / v.sum(1).clamp(min=1)  # [B*C, d]
        out["text_norm"].append(t.norm(dim=-1).mean().item())
        out["text_var_across_samples"].append((t - t.mean(0, keepdim=True)).norm(dim=-1).mean().item())

    def h_prob(mod, inp, outp):
        if not rec["on"]:
            return
        out["psi"].append(outp.detach().flatten().cpu())

    def h_mtb(mod, inp, outp):
        if not rec["on"]:
            return
        llm, tm = inp
        sa, _ = mod.self_attn(llm, llm, llm)
        ca, _ = mod.cross_attn(query=tm, key=sa, value=sa)
        if hasattr(mod, "text_norm"):
            ca = mod.text_gate * mod.text_norm(ca)
        out["cross"].append(norm(ca)); out["time_mask"].append(norm(tm))

    hooks += [m.transformer_block.register_forward_hook(h_tb, with_kwargs=True),
              m.prob_net.register_forward_hook(h_prob),
              m.multimodal_Transformer_Block.register_forward_hook(h_mtb)]

    # 反事实: 同一 batch、同一随机种子下, 只改变文本相关输入, 看输出变化多大
    zero_llm = {"on": False}

    def pre_mtb(mod, args):
        if zero_llm["on"]:
            return (torch.zeros_like(args[0]), args[1])

    hooks.append(m.multimodal_Transformer_Block.register_forward_pre_hook(pre_mtb))
    dot = m.tokenizer(".", max_length=1024, padding="max_length", truncation=True, return_tensors="pt")

    def run(x, ids, am):
        torch.manual_seed(0)  # 固定遮盖和 gumbel 采样的随机性
        o, _, _, _ = m(x, ids, am)
        return o

    cmp = {"dot": [], "shuffle": [], "zero_llm": []}
    with torch.no_grad():
        for i, (x, ids, am, _) in enumerate(self.thre_loader):
            if i >= N_BATCH:
                break
            x, ids, am = x.float().to(dev), ids.float().to(dev), am.float().to(dev)
            rec["on"] = True
            base = run(x, ids, am)
            rec["on"] = False
            s_base = ((x - base) ** 2).mean(-1).flatten()
            B = x.shape[0]
            dids = dot["input_ids"].to(dev).float().expand(B, -1)
            dam = dot["attention_mask"].to(dev).float().expand(B, -1)
            perm = torch.roll(torch.arange(B, device=dev), 1)
            variants = {"dot": run(x, dids, dam), "shuffle": run(x, ids[perm], am[perm])}
            zero_llm["on"] = True
            variants["zero_llm"] = run(x, ids, am)
            zero_llm["on"] = False
            for k, o in variants.items():
                s = ((x - o) ** 2).mean(-1).flatten()
                rel = (o - base).norm().item() / (x - base).norm().item()
                corr = np.corrcoef(s_base.cpu().numpy(), s.cpu().numpy())[0, 1]
                cmp[k].append((rel, corr))
    for h in hooks:
        h.remove()

    psi = torch.cat(out["psi"]).numpy()
    print("\n========== TEXT DIAGNOSIS ==========", flush=True)
    print(f"[压缩器保留概率 psi] mean={psi.mean():.3f} min={psi.min():.3f} max={psi.max():.3f} "
          f"P(psi>0.5)={(psi > 0.5).mean():.2%}  (目标 r={self.config.r})")
    print(f"[跨视图融合] |内生 prompt|={np.mean(out['prompt']):.3f}  "
          f"|外生文本经 attention 后|={np.mean(out['text_attn']):.3f}")
    print(f"[外生文本特征] |text|={np.mean(out['text_norm']):.3f}  "
          f"样本间差异 |text-均值|={np.mean(out['text_var_across_samples']):.4f}")
    print(f"[重建] |被遮盖的时间特征(残差)|={np.mean(out['time_mask']):.3f}  "
          f"|文本经 cross-attn 后|={np.mean(out['cross']):.3f}")
    for k, name in [("dot", "外生文本换成 '.'"), ("shuffle", "外生文本在 batch 内错位"),
                    ("zero_llm", "整个文本分支(内生+外生)置零")]:
        a = np.array(cmp[k])
        print(f"[反事实: {name}] 输出变化/重建误差={a[:, 0].mean():.3%}  "
              f"异常分数相关系数={a[:, 1].mean():.4f}")
    if hasattr(m.multimodal_Transformer_Block, "text_gate"):
        print(f"[重建门控 text_gate] {m.multimodal_Transformer_Block.text_gate.item():.4f} (初值 1)")
    for k, w0 in m._w0.items():
        w = getattr(m, k).weight.detach().cpu()
        print(f"[权重训练前后变化] {k}: shape={tuple(w.shape)} 相对变化={(w - w0).norm() / w0.norm():.4%}")
    sys.stdout.flush()
    os._exit(0)


W.detect_multi_label = diagnose

cmd = shlex.split(open(sys.argv[1]).read().strip())
args = cmd[1:]
args[args.index("--save-path") + 1] = "diag_tmp"
sys.argv = args
runpy.run_path(args[0], run_name="__main__")

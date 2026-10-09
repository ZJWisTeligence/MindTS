"""[临时排查] 外生文本消融: 把每个样本的外生文本替换成 ".", 其余完全不变, 跑完整流程.
内生文本 (模型内部根据时序生成的统计描述) 保持不变.
用法: MINDTS_SEED=<种子> python repro/ablate_notext.py <官方脚本路径> <结果子目录>
不修改任何项目文件, 只在本进程内给 MINDTSModel.forward 打补丁.
"""
import os
import runpy
import shlex
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from ts_benchmark.baselines.MindTS.models import MindTS_model as M  # noqa: E402

_orig = M.MINDTSModel.forward
_cache = {}


def no_text(self, x_enc_time, x_enc_input_ids, x_enc_attention_mask):
    B, L = x_enc_input_ids.shape
    if L not in _cache:
        tok = self.tokenizer(".", max_length=L, padding="max_length", truncation=True, return_tensors="pt")
        _cache[L] = (tok["input_ids"], tok["attention_mask"])
    ids, mask = _cache[L]
    ids = ids.to(x_enc_input_ids.device, x_enc_input_ids.dtype).expand(B, L)
    mask = mask.to(x_enc_attention_mask.device, x_enc_attention_mask.dtype).expand(B, L)
    return _orig(self, x_enc_time, ids, mask)


M.MINDTSModel.forward = no_text

script, out = sys.argv[1], sys.argv[2]
cmd = shlex.split(open(script).read().strip())
args = cmd[1:]
args[args.index("--save-path") + 1] = out
sys.argv = args
runpy.run_path(args[0], run_name="__main__")

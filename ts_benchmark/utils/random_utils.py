import os
import random
from typing import Optional, NoReturn

import numpy as np
import torch


# [复现排查] 可用环境变量 MINDTS_SEED 覆盖默认种子 2021, 不设置时行为与原版一致
_DEFAULT_SEED = int(os.environ.get("MINDTS_SEED", 2021))


def fix_random_seed(seed: Optional[int] = _DEFAULT_SEED) -> NoReturn:
    if seed is None:
        return

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def fix_all_random_seed(seed: Optional[int] = 2021) -> NoReturn:
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.cuda.manual_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(1)

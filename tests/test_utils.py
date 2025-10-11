# tests/test_utils.py
from src.utils.logging import init_logging, get_logger
from src.utils.checkpoints import save_checkpoint_safe, load_checkpoint_safe
from src.utils.seed import set_global_seed
import torch.nn as nn

def test_logging_and_ckpt():
    init_logging("./logs", run_name="utils_test")
    logger = get_logger("utils.test")
    logger.info("hello world")

    class Tiny(nn.Module):
        def __init__(self): super().__init__()
        def forward(self, x): return x

    m = Tiny()
    save_checkpoint_safe(m, "./checkpoints/utils_test", "step_1", save_total_limit=2)
    tag = load_checkpoint_safe("./checkpoints/utils_test")
    assert tag == "step_1"
"""
    BERT 剪枝路线专属配置
"""

import torch
from src.bert.config import BertConfig


class PruneConfig(BertConfig):
    """BERT 权重剪枝方案配置"""

    def __init__(self):
        super().__init__()
        # 覆盖设备：剪枝不需要强制 CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        # 剪枝专属路径
        self.bert_prune_path = self.models_dir / "bert_prune.pt"


if __name__ == "__main__":
    config = PruneConfig()
    print(f"device: {config.device}")
    print(f"bert_prune_path: {config.bert_prune_path}")

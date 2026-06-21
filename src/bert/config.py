"""
    BERT 路线专属配置 — BERT-base-chinese 微调
"""

import torch
from src.common.base_config import BaseConfig


class BertConfig(BaseConfig):
    """BERT 微调方案配置"""

    def __init__(self):
        super().__init__()

        # ── BERT 专属模型路径 ──
        self.bert_model_path = self.models_dir / "bert_model.pt"
        self.bert_quantization_path = self.models_dir / "bert_quantization.pt"

        # ── 设备：动态量化仅支持 CPU ──
        self.device = torch.device("cpu")

        # ── 超参数 ──
        self.epochs = 5
        self.batch_size = 64
        self.lr = 1e-5
        self.max_len = 32
        self.weight_decay = 1e-2


if __name__ == "__main__":
    config = BertConfig()
    print(f"bert_model_path: {config.bert_model_path}")
    print(f"bert_path: {config.bert_path}")
    print(f"device: {config.device}")
    print(f"epochs: {config.epochs}")

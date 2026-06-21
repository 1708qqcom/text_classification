"""
    BERT 蒸馏路线专属配置
"""

import torch
from src.bert.config import BertConfig


class DistillConfig(BertConfig):
    """BERT 知识蒸馏方案配置"""

    def __init__(self):
        super().__init__()
        # 覆盖设备：蒸馏训练不需要强制 CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        # 覆盖超参数
        self.epochs = 10
        self.lr = 1e-4

        # 蒸馏专属路径
        self.student_model_path = self.models_dir / "student_model.pt"

        # 蒸馏参数
        self.alpha = 0.7   # 蒸馏损失权重
        self.T = 4.0       # 温度系数


if __name__ == "__main__":
    config = DistillConfig()
    print(f"device: {config.device}")
    print(f"student_model_path: {config.student_model_path}")
    print(f"alpha: {config.alpha}, T: {config.T}")

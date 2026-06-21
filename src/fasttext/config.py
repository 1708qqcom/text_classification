"""
    FastText 路线专属配置
"""

from src.common.base_config import BaseConfig


class FastTextConfig(BaseConfig):
    """FastText 方案配置"""

    def __init__(self):
        super().__init__()
        # fasttext_model_path 已在 BaseConfig 中定义
        # 各变体通过 replace(".bin", "_char_1.bin") 等方法产生具体路径


if __name__ == "__main__":
    config = FastTextConfig()
    print(f"fasttext_model_path: {config.fasttext_model_path}")
    print(f"char_train_path: {config.char_train_path}")
    print(f"word_train_path: {config.word_train_path}")

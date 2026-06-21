"""
    RF 路线专属配置 — TF-IDF + 随机森林
"""

from src.common.base_config import BaseConfig


class RFConfig(BaseConfig):
    """TF-IDF + 随机森林方案配置"""

    def __init__(self):
        super().__init__()
        # RF 专属路径已在 BaseConfig 中定义：rf_model_path, tfidf_model_path
        # 此处仅做显式声明，如需覆盖可在此修改


if __name__ == "__main__":
    config = RFConfig()
    print(f"data_dir: {config.data_dir}")
    print(f"models_dir: {config.models_dir}")
    print(f"rf_model_path: {config.rf_model_path}")
    print(f"tfidf_model_path: {config.tfidf_model_path}")
    print(f"id2class: {config.id2class}")

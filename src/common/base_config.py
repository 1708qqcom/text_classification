"""
    公共基础配置类 — 所有技术路线共享的数据路径、API 配置、类别映射。
    各路线通过继承此类，仅声明自己专属的模型路径和超参数。
"""

import os
import torch
from pathlib import Path


class BaseConfig:
    """所有技术路线共享的基础配置"""

    def __init__(self):
        # ── 自动定位项目根目录（基于本文件位置，不依赖 cwd） ──
        self.project_root = Path(__file__).resolve().parent.parent.parent

        # ── 目录 ──
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        os.makedirs(self.models_dir, exist_ok=True)

        # ── 原始数据 ──
        self.train_raw_path = self.data_dir / "train_raw.txt"
        self.test_raw_path = self.data_dir / "test_raw.txt"
        self.dev_raw_path = self.data_dir / "dev_raw.txt"

        # ── 去重后数据 ──
        self.train_path = self.data_dir / "train.txt"
        self.test_path = self.data_dir / "test.txt"
        self.dev_path = self.data_dir / "dev.txt"

        # ── 停用词 & 类别 ──
        self.stopwords_path = self.data_dir / "stopwords.txt"
        self.class_path = self.data_dir / "class.txt"

        # ── 中间产物（RF + FastText 共用） ──
        self.process_train_path = self.data_dir / "process_train.csv"
        self.process_test_path = self.data_dir / "process_test.csv"
        self.process_dev_path = self.data_dir / "process_dev.csv"

        # ── FastText 中间产物 ──
        self.char_train_path = self.data_dir / "char_train.txt"
        self.char_test_path = self.data_dir / "char_test.txt"
        self.char_dev_path = self.data_dir / "char_dev.txt"
        self.word_train_path = self.data_dir / "word_train.txt"
        self.word_test_path = self.data_dir / "word_test.txt"
        self.word_dev_path = self.data_dir / "word_dev.txt"

        # ── 预训练模型 ──
        self.bert_path = self.data_dir / "bert-base-chinese"

        # ── 模型保存路径（通用） ──
        self.rf_model_path = self.models_dir / "rf_model.pkl"
        self.tfidf_model_path = self.models_dir / "tfidf_model.pkl"
        self.fasttext_model_path = self.models_dir / "fasttext_model.bin"

        # ── 预测结果保存目录 ──
        self.result_dir = self.project_root / "results"
        os.makedirs(self.result_dir, exist_ok=True)
        self.model_predict_result = self.result_dir / "model_predict_result.txt"

        # ── API 配置 ──
        self.api_host = "127.0.0.1"
        self.api_port = 5000

        # ── 设备 ──
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        # ── 类别索引 → 名称映射 ──
        with open(self.class_path, "r", encoding="utf-8") as f:
            self.id2class = {i: line.strip() for i, line in enumerate(f)}


if __name__ == "__main__":
    config = BaseConfig()
    print(f"project_root: {config.project_root}")
    print(f"data_dir: {config.data_dir}")
    print(f"models_dir: {config.models_dir}")
    print(f"class_path: {config.class_path}")
    print(f"api_host: {config.api_host}")
    print(f"api_port: {config.api_port}")
    print(f"device: {config.device}")
    print(f"id2class: {config.id2class}")

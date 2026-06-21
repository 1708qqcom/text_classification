"""
    数据预处理 — jieba 分词，保存为 CSV
"""

import pandas as pd
import jieba
from src.rf.config import RFConfig
import matplotlib.pyplot as plt

config = RFConfig()


def process_data(input_file, output_file):
    """
    数据预处理函数，对原始数据集进行 jieba 分词，并保存为 CSV 文件。

    Args:
        input_file: 原始数据集文件，txt 格式，每行: text \\t label
        output_file: 处理后 CSV 文件路径，包含: text, label, words, seq_len
    """
    # 1. 读取原始文件
    data = pd.read_csv(input_file, sep="\t", names=["text", "label"])

    # 2. jieba 分词
    data["words"] = data["text"].apply(lambda x: " ".join(jieba.lcut(x)[:20]))

    # 3. 序列长度
    data["seq_len"] = data["text"].apply(lambda x: len(jieba.lcut(x)))

    # 4. 可视化序列长度分布
    data["seq_len"].hist()
    plt.show()
    print(data["seq_len"].describe())

    # 5. 保存处理后的数据
    data.to_csv(output_file, index=False)
    print(f"保存处理后的数据到 CSV 文件: {output_file}")


if __name__ == "__main__":
    process_data(config.train_path, config.process_train_path)
    process_data(config.test_path, config.process_test_path)
    process_data(config.dev_path, config.process_dev_path)

"""
    EDA (Exploratory Data Analysis) — 探索性数据分析
    查看标签数量分布、句子长度分布
"""

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from src.common.base_config import BaseConfig

# 1. 初始化配置类
config = BaseConfig()

# 2. 读取数据集
train_data = pd.read_csv(config.train_path, sep="\t", names=["text", "label"])
dev_data = pd.read_csv(config.dev_path, sep="\t", names=["text", "label"])
test_data = pd.read_csv(config.test_path, sep="\t", names=["text", "label"])
print(train_data)

# 建议优先查看训练集的数据，来选择 seq_len
# 也可以查看验证集的数据
# 不建议直接查看测试集的数据，并用它作为选择 seq_len 的标准

# 3. 统计标签分布
label_counts = Counter(train_data["label"])
print(f"label_counts: {label_counts}")
for label, count in label_counts.items():
    print(f"label: {label}, count: {count}")

# 4. 绘制标签分布直方图
train_data["label"].hist()
plt.show()

# 5. 统计句子长度
train_data["text_len"] = train_data["text"].apply(lambda x: len(x))
print(train_data["text_len"].describe())

# 6. 绘制句子长度分布直方图
train_data["text_len"].hist()
plt.show()

"""
    随机森林模型 — 测试集评估
    工作流：加载数据 → 加载模型和向量化器 → TF-IDF 向量化 → 模型评估
"""

import pandas as pd
import pickle
from src.rf.config import RFConfig
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)

config = RFConfig()

# 1. 加载模型和向量化器
with open(config.rf_model_path, 'rb') as f:
    rf_model = pickle.load(f)
with open(config.tfidf_model_path, 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# 2. 加载 dev 数据
dev_data = pd.read_csv(config.process_dev_path)
print(dev_data.shape)

words = dev_data['words']
labels = dev_data['label']

print(f"TF-IDF 向量化器词表大小：{len(tfidf_vectorizer.vocabulary_)}")
features = tfidf_vectorizer.transform(words)
print(f"features 维度：{features.shape}")

# 3. 模型预测与评估
y_pred = rf_model.predict(features)

print(f"准确率：{accuracy_score(labels, y_pred)}")
print(f"精确率 macro：{precision_score(labels, y_pred, average='macro')}")
print(f"召回率 macro：{recall_score(labels, y_pred, average='macro')}")
print(f"F1-score macro：{f1_score(labels, y_pred, average='macro')}")
print(f"分类评估报告：\n{classification_report(labels, y_pred)}")
print(f"混淆矩阵：\n{confusion_matrix(labels, y_pred)}")

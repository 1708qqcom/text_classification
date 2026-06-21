"""
    训练随机森林模型 + TF-IDF 向量化器
    工作流：加载数据 → TF-IDF 特征提取 → 训练 → 评估 → 保存模型
"""

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from src.rf.config import RFConfig

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)

config = RFConfig()

# 1. 加载数据集
train_data = pd.read_csv(config.process_train_path)[:20000]
print(f"训练集样本量：{len(train_data)}")

# 2. 文本数值化：TF-IDF
words = train_data["words"]
labels = train_data["label"]

with open(config.stopwords_path, "r", encoding="utf-8") as f:
    stopwords = [line.strip() for line in f if line.strip()]
print(f"stopwords 数量: {len(stopwords)}")

tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords)
features = tfidf_vectorizer.fit_transform(words)
print(f"features.shape: {features.shape}")
print(f"词表大小: {len(tfidf_vectorizer.vocabulary_)}")

# 3. 训练随机森林模型
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=7
)
print(f"训练集样本量：{x_train.shape}")
print(f"测试集样本量：{x_test.shape}")

rf_model = RandomForestClassifier(n_estimators=100, random_state=7)
rf_model.fit(x_train, y_train)

# 4. 模型评估
y_pred = rf_model.predict(x_test)
print(f"准确率: {accuracy_score(y_test, y_pred)}")
print(f"精确率: {precision_score(y_test, y_pred, average='macro')}")
print(f"召回率: {recall_score(y_test, y_pred, average='macro')}")
print(f"F1-score: {f1_score(y_test, y_pred, average='macro')}")
print(f"分类评估报告:\n{classification_report(y_test, y_pred)}")
print(f"混淆矩阵:\n{confusion_matrix(y_test, y_pred)}")

# 5. 保存模型
with open(config.rf_model_path, "wb") as f:
    pickle.dump(rf_model, f)
print(f"保存随机森林模型: {config.rf_model_path}")

with open(config.tfidf_model_path, "wb") as f:
    pickle.dump(tfidf_vectorizer, f)
print(f"保存 TF-IDF 向量化器: {config.tfidf_model_path}")

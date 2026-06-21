"""
    FastText 训练 — 词级分词 + 手动调参
"""

import fasttext
from src.fasttext.config import FastTextConfig

config = FastTextConfig()

model = fasttext.train_supervised(
    input=config.word_train_path,
    dim=20,
    wordNgrams=2,
    epoch=5,
    lr=0.1,
)

model.save_model(config.fasttext_model_path.replace(".bin", "_word_1.bin"))
print(f'保存模型到: {config.fasttext_model_path.replace(".bin", "_word_1.bin")}')

result = model.test(config.word_test_path)
print(f"评估结果(样本数, 精确率P, 召回率R): {result}")

f1_score = 2 * result[1] * result[2] / (result[1] + result[2])
print(f"F1_score: {f1_score}")

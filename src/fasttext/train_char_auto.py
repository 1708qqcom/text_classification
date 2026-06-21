"""
    FastText 训练 — 字符级分词 + 自动调参
"""

import fasttext
from src.fasttext.config import FastTextConfig

config = FastTextConfig()

model = fasttext.train_supervised(
    input=config.char_train_path,
    autotuneValidationFile=config.char_dev_path,
    autotuneDuration=300,  # 自动调参时间（秒）
)

model.save_model(config.fasttext_model_path.replace(".bin", "_char_2.bin"))
print(f'保存模型到: {config.fasttext_model_path.replace(".bin", "_char_2.bin")}')

result = model.test(config.char_test_path)
print(f"评估结果(样本数, 精确率P, 召回率R): {result}")

f1_score = 2 * result[1] * result[2] / (result[1] + result[2])
print(f"F1_score: {f1_score}")

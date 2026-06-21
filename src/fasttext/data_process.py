"""
    数据预处理 — 将原始文本转为 FastText 输入格式
    标签前缀为 __label__，分隔符为空格
"""

import jieba
from src.fasttext.config import FastTextConfig

config = FastTextConfig()


def process_data(input_path, output_path, is_char=True):
    """
    数据处理函数，加载原始文本，生成 FastText 输入格式并保存。

    Args:
        input_path: 原始文件路径
        output_path: 处理后的文件路径
        is_char: True 按字分词，False 按 jieba 词分词
    """
    with open(input_path, "r", encoding="utf-8") as f:
        with open(output_path, "w", encoding="utf-8") as fw:
            for line in f:
                line = line.strip()
                text, label = line.split("\t")
                label_name = config.id2class[int(label)]

                if is_char:
                    text_split = " ".join(list(text))
                else:
                    text_split = " ".join(jieba.lcut(text))

                ft_line = f"__label__{label_name} {text_split}\n"
                fw.write(ft_line)


if __name__ == "__main__":
    # 按字分词
    process_data(config.train_path, config.char_train_path, is_char=True)
    process_data(config.dev_path, config.char_dev_path, is_char=True)
    process_data(config.test_path, config.char_test_path, is_char=True)
    # 按词分词
    process_data(config.train_path, config.word_train_path, is_char=False)
    process_data(config.dev_path, config.word_dev_path, is_char=False)
    process_data(config.test_path, config.word_test_path, is_char=False)

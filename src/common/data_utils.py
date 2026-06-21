"""
    数据处理工具函数
"""

import os


def remove_duplicates(input_file, output_file):
    """
    读取文件去重并保存。

    Args:
        input_file: 输入文件路径
        output_file: 去重后输出文件路径
    """
    seen = set()
    total_lines = 0
    unique_lines = 0

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            total_lines += 1
            if line not in seen:
                outfile.write(line)
                seen.add(line)
                unique_lines += 1

    print(f"去重前总行数: {total_lines}")
    print(f"去重后保留行数: {unique_lines}")
    print(f"重复行数: {total_lines - unique_lines}")
    print("-" * 30)


if __name__ == "__main__":
    from src.common.base_config import BaseConfig
    config = BaseConfig()
    remove_duplicates(config.train_raw_path, config.train_path)
    remove_duplicates(config.test_raw_path, config.test_path)
    remove_duplicates(config.dev_raw_path, config.dev_path)

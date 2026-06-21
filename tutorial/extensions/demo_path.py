"""
演示
    使用Path(__file__)获取当前文件路径
    比如D:\tmf_project_shanghai_AI_2\08-live_code\01-data\10_扩展_演示Path.py

"""

from pathlib import Path  # 路径操作模块
import os

# 获取当前文件路径
print(__file__)
print(Path(__file__))

# 获取当前文件的父路径
print(Path(__file__).parent)
print(os.path.dirname(__file__))

# 获取当前文件的父路径的父路径
print(Path(__file__).parent.parent)
print(os.path.dirname(os.path.dirname(__file__)))

# 获取当前文件的父路径的01-data
print(os.path.join(Path(__file__).parent.parent,"01-data"))
print(os.path.join(os.path.dirname(os.path.dirname(__file__)), "01-data"))
print(f"{Path(__file__).parent.parent}\\01-data")


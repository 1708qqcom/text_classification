"""
    BERT Distill Streamlit 前端入口（双按钮：教师 + 学生对比）
    启动: streamlit run src/distill/streamlit_app.py
"""

from src.distill.config import DistillConfig
from src.serving.streamlit_app import run_streamlit_page

config = DistillConfig()

run_streamlit_page(
    buttons=[
        {"label": "BERT预测-教师模型", "url": f"http://{config.api_host}:{config.api_port}/predict1"},
        {"label": "BERT预测-学生模型", "url": f"http://{config.api_host}:{config.api_port}/predict2"},
    ],
    title="投满分项目"
)

"""
    BERT Prune Streamlit 前端入口
    启动: streamlit run src/prune/streamlit_app.py
"""

from src.prune.config import PruneConfig
from src.serving.streamlit_app import run_streamlit_page

config = PruneConfig()
url = f"http://{config.api_host}:{config.api_port}/predict"

run_streamlit_page(
    buttons=[{"label": "BERT剪枝预测", "url": url}],
    title="投满分项目"
)

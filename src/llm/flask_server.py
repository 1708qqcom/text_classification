"""
    LLM Flask API 服务入口
"""

from src.llm.predict import predict_fun
from src.common.base_config import BaseConfig
from src.serving.flask_app import create_flask_app

config = BaseConfig()
app = create_flask_app(
    predict_endpoints=[("/predict", "LLM预测", predict_fun)],
    title="LLM预测服务"
)

if __name__ == '__main__':
    app.run(host=config.api_host, port=config.api_port, debug=True)

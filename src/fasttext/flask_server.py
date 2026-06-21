"""
    FastText Flask API 服务入口
"""

from src.fasttext.predict import predict_fun
from src.fasttext.config import FastTextConfig
from src.serving.flask_app import create_flask_app

config = FastTextConfig()
app = create_flask_app(
    predict_endpoints=[("/predict", "FastText预测", predict_fun)],
    title="FastText预测服务"
)

if __name__ == '__main__':
    app.run(host=config.api_host, port=config.api_port, debug=True)

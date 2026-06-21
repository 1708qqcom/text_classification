"""
    RF Flask API 服务入口
    启动: python -m src.rf.flask_server
"""

from src.rf.predict import predict_fun
from src.rf.config import RFConfig
from src.serving.flask_app import create_flask_app

config = RFConfig()
app = create_flask_app(
    predict_endpoints=[("/predict", "RF预测", predict_fun)],
    title="RF预测服务"
)

if __name__ == '__main__':
    app.run(host=config.api_host, port=config.api_port, debug=True)

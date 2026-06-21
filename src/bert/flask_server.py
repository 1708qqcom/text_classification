"""
    BERT Flask API 服务入口
"""

from src.bert.predict import predict_fun
from src.bert.config import BertConfig
from src.serving.flask_app import create_flask_app

config = BertConfig()
app = create_flask_app(
    predict_endpoints=[("/predict", "BERT预测", predict_fun)],
    title="BERT预测服务"
)

if __name__ == '__main__':
    app.run(host=config.api_host, port=config.api_port, debug=True)

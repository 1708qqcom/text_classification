"""
    BERT Prune Flask API 服务入口
"""

from src.prune.predict import predict_fun
from src.prune.config import PruneConfig
from src.serving.flask_app import create_flask_app

config = PruneConfig()
app = create_flask_app(
    predict_endpoints=[("/predict", "BERT剪枝预测", predict_fun)],
    title="BERT剪枝预测服务"
)

if __name__ == '__main__':
    app.run(host=config.api_host, port=config.api_port, debug=True)

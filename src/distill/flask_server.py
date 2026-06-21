"""
    BERT Distill Flask API 服务入口（双路由：教师 + 学生）
"""

from src.distill.predict import predict_fun1, predict_fun2
from src.distill.config import DistillConfig
from src.serving.flask_app import create_flask_app

config = DistillConfig()
app = create_flask_app(
    predict_endpoints=[
        ("/predict1", "教师模型", predict_fun1),
        ("/predict2", "学生模型", predict_fun2),
    ],
    title="BERT蒸馏预测服务"
)

if __name__ == '__main__':
    app.run(host=config.api_host, port=config.api_port, debug=True)

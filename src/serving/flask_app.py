"""
    通用 Flask API 工厂函数
    支持单路由和多路由模式（如蒸馏路线需要两个预测端点）
"""

from flask import Flask, request, jsonify


def create_flask_app(predict_endpoints, title="预测服务", host="127.0.0.1", port=5000):
    """
    创建 Flask 应用。

    Args:
        predict_endpoints: list of (route, label, predict_fn)
            例: [("/predict", "RF预测", rf_predict)]
            例: [("/predict1", "教师模型", teacher_predict),
                 ("/predict2", "学生模型", student_predict)]
        title: 应用名称
        host: 监听地址
        port: 监听端口

    Returns:
        Flask app 实例（未启动）
    """
    app = Flask(title)

    for route, label, predict_fn in predict_endpoints:
        # 用闭包捕获当前 predict_fn
        def make_handler(fn=predict_fn):
            def handler():
                data = request.get_json()
                print(f"[{label}] data: {data}")
                result = fn(data)
                return jsonify(result)
            return handler

        app.route(route, methods=['POST'], endpoint=route.replace('/', '_'))(make_handler())

    return app


def run_flask_app(app, host="127.0.0.1", port=5000, debug=True):
    """启动 Flask 应用"""
    app.run(host=host, port=port, debug=debug)

"""
演示
    flask框架入门操作，创建一个简单的API服务
需要掌握:
    安装flask: pip install flask
    app = Flask(__name__)   # 创建Flask应用
    @app.route('/', methods=['GET'])    # 创建路由
    def predict(): 视图函数
"""
from flask import Flask, Response

# 1.创建Flask应用
app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    html = """
    <!doctype html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>投满分项目 - 首页</title>
        <style>
            * {
                box-sizing: border-box;
            }
            body {
                margin: 0;
                min-height: 100vh;
                font-family: "Microsoft YaHei", Arial, sans-serif;
                background: linear-gradient(135deg, #eef2ff, #f8fafc);
                display: flex;
                justify-content: center;
                align-items: center;
                color: #1f2937;
            }
            .card {
                width: min(92vw, 560px);
                background: #ffffff;
                border-radius: 18px;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
                padding: 46px 38px;
                text-align: center;
            }
            h1 {
                margin: 0 0 14px;
                font-size: 34px;
                color: #1e40af;
            }
            p {
                margin: 0 0 28px;
                color: #4b5563;
                font-size: 16px;
            }
            .btn {
                display: inline-block;
                text-decoration: none;
                background: #2563eb;
                color: #ffffff;
                padding: 12px 28px;
                border-radius: 999px;
                font-weight: 600;
                transition: all 0.2s ease;
            }
            .btn:hover {
                background: #1d4ed8;
                transform: translateY(-1px);
                box-shadow: 0 8px 16px rgba(37, 99, 235, 0.25);
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>欢迎来到投满分项目</h1>
            <p>点击下方按钮，开始查看图片内容。</p>
            <a class="btn" href="/image">开始</a>
        </div>
    </body>
    </html>
    """
    return Response(html, mimetype="text/html")


@app.route("/image", methods=["GET"])
def show_image():
    html = """
    <!doctype html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>投满分项目 - 图片展示</title>
        <style>
            * {
                box-sizing: border-box;
            }
            body {
                margin: 0;
                min-height: 100vh;
                font-family: "Microsoft YaHei", Arial, sans-serif;
                background: linear-gradient(145deg, #f0f9ff, #ecfeff);
                display: flex;
                justify-content: center;
                align-items: center;
                color: #1f2937;
                padding: 20px;
            }
            .card {
                width: min(95vw, 900px);
                background: #ffffff;
                border-radius: 18px;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
                padding: 24px;
                text-align: center;
            }
            h2 {
                margin: 6px 0 18px;
                color: #0f766e;
            }
            img {
                max-width: 100%;
                height: auto;
                border-radius: 12px;
                border: 1px solid #e5e7eb;
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
            }
            .back {
                margin-top: 20px;
            }
            .btn {
                display: inline-block;
                text-decoration: none;
                background: #0ea5e9;
                color: #ffffff;
                padding: 10px 22px;
                border-radius: 999px;
                font-weight: 600;
                transition: all 0.2s ease;
            }
            .btn:hover {
                background: #0284c7;
                transform: translateY(-1px);
                box-shadow: 0 8px 16px rgba(2, 132, 199, 0.24);
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>图片展示</h2>
            <img src="/static-image" alt="项目图片" />
            <div class="back">
                <a class="btn" href="/">返回首页</a>
            </div>
        </div>
    </body>
    </html>
    """
    return Response(html, mimetype="text/html")


@app.route("/static-image", methods=["GET"])
def static_image():
    # 复用原有图片返回逻辑
    from flask import send_file

    return send_file("img.jpg")


# 3.启动服务
if __name__ == "__main__":
    # 1.只允许本机访问
    app.run(host="127.0.0.1", port=5000, debug=True)

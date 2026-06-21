"""
演示：
    FastAPI 框架入门操作，创建一个简单的 API 服务
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# 当前脚本所在目录：.../00-test
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="FastAPI入门演示",
    description="FastAPI入门的课堂示例",
    version="1.0.0"
)

# 挂载静态目录，用于访问 img.jpg
# 访问路径示例：/static/img.jpg
app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!doctype html>
    <html lang="zh-CN">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>投满分项目 - 首页</title>
      <style>
        * { box-sizing: border-box; }
        body {
          margin: 0;
          min-height: 100vh;
          display: flex;
          justify-content: center;
          align-items: center;
          font-family: "Microsoft YaHei", Arial, sans-serif;
          background: linear-gradient(135deg, #e0f2fe, #eef2ff);
          color: #0f172a;
        }
        .card {
          width: min(92vw, 560px);
          background: #fff;
          border-radius: 18px;
          box-shadow: 0 12px 28px rgba(15, 23, 42, 0.12);
          padding: 44px 32px;
          text-align: center;
        }
        h1 {
          margin: 0 0 14px;
          font-size: 34px;
          color: #1d4ed8;
        }
        p {
          margin: 0 0 28px;
          color: #475569;
        }
        .btn {
          display: inline-block;
          text-decoration: none;
          background: #2563eb;
          color: #fff;
          padding: 12px 28px;
          border-radius: 999px;
          font-weight: 700;
          transition: .2s;
        }
        .btn:hover {
          background: #1e40af;
          transform: translateY(-1px);
        }
      </style>
    </head>
    <body>
      <div class="card">
        <h1>欢迎来到投满分项目</h1>
        <p>点击开始，进入图片展示页面。</p>
        <a class="btn" href="/image">开始</a>
      </div>
    </body>
    </html>
    """


@app.get("/image", response_class=HTMLResponse)
def image_page():
    return """
    <!doctype html>
    <html lang="zh-CN">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>投满分项目 - 图片展示</title>
      <style>
        * { box-sizing: border-box; }
        body {
          margin: 0;
          min-height: 100vh;
          display: flex;
          justify-content: center;
          align-items: center;
          font-family: "Microsoft YaHei", Arial, sans-serif;
          background: linear-gradient(145deg, #f0f9ff, #f8fafc);
          padding: 20px;
        }
        .card {
          width: min(95vw, 900px);
          background: #fff;
          border-radius: 18px;
          box-shadow: 0 12px 28px rgba(15, 23, 42, 0.12);
          padding: 24px;
          text-align: center;
        }
        h2 {
          margin: 4px 0 16px;
          color: #0f766e;
        }
        img {
          max-width: 100%;
          height: auto;
          border-radius: 12px;
          border: 1px solid #e2e8f0;
          box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        }
        .actions { margin-top: 18px; }
        .btn {
          display: inline-block;
          text-decoration: none;
          background: #0ea5e9;
          color: #fff;
          padding: 10px 22px;
          border-radius: 999px;
          font-weight: 700;
          transition: .2s;
        }
        .btn:hover {
          background: #0284c7;
          transform: translateY(-1px);
        }
      </style>
    </head>
    <body>
      <div class="card">
        <h2>图片展示</h2>
        <img src="/static/img.jpg" alt="img.jpg" />
        <div class="actions">
          <a class="btn" href="/">返回首页</a>
        </div>
      </div>
    </body>
    </html>
    """


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

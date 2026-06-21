# 投满分 — 中文文本多分类全链路实战

基于中文新闻短文本的 **10 分类** 任务，完整覆盖从传统机器学习到深度学习到大语言模型的 **6 条技术路线**，每条路线均实现：数据处理 → 模型训练 → 评估 → 模型压缩 → API 部署 → Web 前端。

## 任务定义

| 项目 | 说明 |
|------|------|
| 任务类型 | 单标签多分类（10 类） |
| 输入 | 中文短文本 / 新闻标题 |
| 输出 | 类别名称 |
| 类别 | finance, realty, stocks, education, science, society, politics, sports, game, entertainment |
| 数据格式 | `text\tlabel_index`（每行一条样本） |

## 项目结构

```
投满分/
├── pyproject.toml              # 项目元数据 + 依赖声明
├── requirements.txt            # pip 依赖清单
├── .gitignore                  # Git 忽略规则
├── .env.example                # API Key 模板（不含真实值）
├── README.md
├── CLAUDE.md
│
├── data/                       # 纯数据资产（不含代码）
│   ├── train_raw.txt           # 原始训练集
│   ├── test_raw.txt            # 原始测试集
│   ├── dev_raw.txt             # 原始验证集
│   ├── train.txt / test.txt / dev.txt  # 去重后数据
│   ├── class.txt               # 类别名称映射
│   ├── stopwords.txt           # 停用词表（800+ 词）
│   ├── bert-base-chinese/      # BERT 预训练权重（HuggingFace）
│   └── (process_*.csv, char_*.txt, word_*.txt 等中间产物)
│
├── models/                     # 训练产出统一存放（整个目录 .gitignore）
│   ├── rf_model.pkl / tfidf_model.pkl
│   ├── fasttext_model_*.bin
│   ├── bert_model.pt / bert_quantization.pt / bert_prune.pt
│   └── student_model.pt
│
├── src/                        # 源码
│   ├── common/                 # 公共基础设施
│   │   ├── base_config.py      # 统一的 BaseConfig（继承体系消除重复）
│   │   ├── data_utils.py       # 数据工具（去重等）
│   │   └── eda.py              # 探索性数据分析
│   │
│   ├── serving/                # 统一的 API 服务层
│   │   ├── flask_app.py        # Flask 工厂函数
│   │   ├── fastapi_app.py      # FastAPI 工厂函数
│   │   ├── streamlit_app.py    # Streamlit 页面组件
│   │   └── api_test.py         # 通用 API 测试客户端
│   │
│   ├── rf/                     # 路线一：TF-IDF + 随机森林
│   │   ├── config.py / data_process.py
│   │   ├── train.py / test.py / predict.py
│   │   ├── flask_server.py / fastapi_server.py / streamlit_app.py
│   │
│   ├── fasttext/               # 路线二：FastText（字/词粒度 + 自动调参）
│   │   ├── config.py / data_process.py / predict.py
│   │   ├── train_char_default.py / train_char_auto.py
│   │   ├── train_word_default.py / train_word_auto.py
│   │   ├── flask_server.py / fastapi_server.py / streamlit_app.py
│   │
│   ├── bert/                   # 路线三：BERT 微调（蒸馏/剪枝的代码基）
│   │   ├── config.py / dataset.py
│   │   ├── train.py / predict.py / quantization.py
│   │   ├── flask_server.py / fastapi_server.py / streamlit_app.py
│   │
│   ├── distill/                # 路线四：BERT 知识蒸馏
│   │   ├── config.py / train_student.py / predict.py
│   │   ├── flask_server.py / fastapi_server.py / streamlit_app.py
│   │
│   ├── prune/                  # 路线五：BERT 权重剪枝
│   │   ├── config.py / prune.py / predict.py
│   │   ├── flask_server.py / fastapi_server.py / streamlit_app.py
│   │
│   └── llm/                    # 路线六：DeepSeek Prompt 工程
│       ├── api_demo.py / prompt_classify.py / predict.py
│       ├── flask_server.py / fastapi_server.py / streamlit_app.py
│
├── tutorial/                   # 教学/大纲类文档
│   ├── flask/                  # Flask 入门教学
│   ├── fastapi/                # FastAPI 入门教学
│   ├── extensions/             # 扩展演示（Path、List、Zip）
│   └── interview/              # 面试准备文档
│
├── assets/                     # 静态资源
└── tests/                      # 测试目录
```

## 技术路线演进

```
传统 ML                  深度学习                  大模型 + 推理优化
──────────────────────────────────────────────────────────────────
TF-IDF + RF ──→ FastText ──→ BERT ──→ BERT 蒸馏 / 剪枝
                                        │
                                        └──→ DeepSeek Prompt 工程
```

### 方案对比

| 维度 | TF-IDF + RF | FastText | BERT 微调 | BERT 蒸馏 | BERT 剪枝 | DeepSeek LLM |
|------|------------|----------|-----------|-----------|-----------|--------------|
| 核心算法 | 词袋 + 集成树 | n-gram + 浅层网络 | Transformer + Fine-tune | Teacher-Student KD | 权重剪枝 | Prompt 工程 |
| 参数量 | — | ~2M | ~110M | ~30M | ~55M | API 调用 |
| 训练速度 | 快（分钟级） | 快（分钟级） | 中等（GPU 小时级） | 中等 | 中等 | — |
| 推理延迟 | < 5ms | < 1ms | ~50ms | ~20ms | ~30ms | ~500ms |
| 中文理解 | 弱（词袋无顺序） | 一般（n-gram） | 强（语义理解） | 较强 | 较强 | 最强 |
| 适用场景 | Baseline 快速验证 | 轻量级上线 | 精排 / 高精度需求 | 端侧部署 | 端侧部署 | 冷启动 / 少样本 |
| 部署方式 | pickle + Flask/FastAPI | bin + Flask/FastAPI | torch + Flask/FastAPI | torch + Flask/FastAPI | torch + Flask/FastAPI | OpenAI SDK |

### 模型优化技术

| 技术 | 目标 | 方法 | 效果预期 |
|------|------|------|---------|
| 动态量化 (DQ) | 推理加速 / 减体积 | Linear 层 FP32 → INT8 | 体积 ~4×↓，延迟 ~2×↓ |
| 知识蒸馏 (KD) | 压缩模型 | BERT Teacher → 小模型 Student | 体积 ~3×↓，精度损失 < 2% |
| 权重剪枝 (Pruning) | 稀疏化加速 | 按比例裁剪低权重连接 | 体积 ~2×↓，精度损失 < 3% |

## 快速开始

### 环境要求

```bash
Python >= 3.10
pip install -r requirements.txt
```

### 1. 数据预处理

```bash
# 去重数据
python -m src.common.data_utils

# 探索性数据分析
python -m src.common.eda
```

### 2. 训练模型

```bash
# 路线一：TF-IDF + 随机森林
python -m src.rf.data_process    # jieba 分词
python -m src.rf.train           # 训练 & 评估 & 保存模型
python -m src.rf.test            # 测试集评估

# 路线二：FastText
python -m src.fasttext.data_process
python -m src.fasttext.train_char_default
python -m src.fasttext.train_char_auto
python -m src.fasttext.train_word_default
python -m src.fasttext.train_word_auto

# 路线三：BERT 微调
python -m src.bert.train

# BERT 量化（训练后）
python -m src.bert.quantization

# 路线四：BERT 蒸馏
python -m src.distill.train_student

# 路线五：BERT 剪枝
python -m src.prune.prune

# 路线六：DeepSeek API
# 配置 .env: DEEPSEEK_API_KEY=your_key
python -m src.llm.predict
```

### 3. 启动 API 服务

```bash
# 路线一（RF）
python -m src.rf.fastapi_server     # http://127.0.0.1:5000/docs

# 路线四（蒸馏，双路由：/predict1 + /predict2）
python -m src.distill.fastapi_server
```

API 接口：

```json
POST /predict
{
  "text": "今天股市大涨，沪指突破3000点"
}

Response:
{
  "text": "今天股市大涨，沪指突破3000点",
  "pred_class": "stocks"
}
```

### 4. 启动 Web 前端

```bash
streamlit run src/rf/streamlit_app.py
```

## Config 继承体系

```
BaseConfig (src/common/base_config.py)
  ├── 数据路径、类别映射、API 配置
  ├── project_root 自动检测（基于文件位置）
  │
  ├── RFConfig (src/rf/config.py)
  ├── FastTextConfig (src/fasttext/config.py)
  ├── BertConfig (src/bert/config.py)       ← 蒸馏/剪枝的基类
  │   ├── DistillConfig (src/distill/config.py)
  │   └── PruneConfig (src/prune/config.py)
  └── LLM 路线直接使用 BaseConfig
```

## 面试中可能被追问的点

### Q1: 为什么选了这 6 种方案而不是只做一种？
> 覆盖了 NLP 文本分类的完整技术谱系：传统 ML baseline → 轻量级深度模型 → 预训练大模型 → LLM API，核心价值在于**可对比的模型选型量化标准**，而不是单点调参。

### Q2: TF-IDF 和 BERT 的效果差距有多大？为什么？
> TF-IDF 丢失了词序和上下文语义，"中国队打败日本队"和"日本队打败中国队"对它来说特征几乎一样。BERT 通过 Self-Attention 捕获了这种差异。

### Q3: 何时用 FastText 而不是 BERT？
> 对延迟敏感（< 1ms）、资源受限（CPU only）、或数据量极大需要快速迭代时。FastText 在短文本分类上的性价比较高。

### Q4: 知识蒸馏具体怎么做的？
> Teacher 是微调好的 BERT-base，Student 是较小的 BERT 变体。用 Teacher 的 soft logits（带温度系数 T）作为软标签训练 Student，结合硬标签的交叉熵损失。

### Q5: 模型上线后如何处理 OOV（未登录词）？
> BERT 使用 WordPiece 子词分词，天然不存在 OOV 问题。FastText 使用 n-gram 子词也能处理。TF-IDF 方案下通过 jieba 分词 + HMM 新词发现缓解。

### Q6: 如果线上 QPS 很高怎么办？
> 1. 模型量化（INT8）降低计算量；2. 批量推理，提高 GPU 利用率；3. 模型蒸馏出更小的模型；4. 多实例 + 负载均衡；5. 考虑用 Triton Inference Server 等专业部署框架。

### Q7: 这个项目里你觉得最难的点是什么？
> BERT 蒸馏：需要在 Teacher soft logits、Student 结构设计、温度参数之间做平衡，同时保证压缩后精度不崩塌。

## License

MIT

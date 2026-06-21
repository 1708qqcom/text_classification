# 中文文本分类 — 多技术路线实战项目

基于同一份中文新闻数据集，覆盖 **TF-IDF + 随机森林 → FastText → BERT 微调 → 模型压缩 → LLM Prompt 工程** 的完整技术演进路线，每条路线均提供训练、预测、Web 服务化的端到端实现。

## 项目结构

```
text_classification/
├── data/                        # 数据集 & 预训练模型
│   ├── train_raw.txt            # 原始训练集（TSV: text \t label）
│   ├── test_raw.txt / dev_raw.txt
│   ├── class.txt                # 10个类别标签
│   ├── stopwords.txt            # 中文停用词表
│   └── bert-base-chinese/       # 本地 BERT 预训练模型
├── models/                      # 训练产出的模型文件
├── results/                     # 预测结果输出
├── src/
│   ├── common/                  # 公共基础配置与工具
│   │   ├── base_config.py       # 所有路线的基类（路径、设备、类别映射）
│   │   ├── data_utils.py        # 数据去重、预处理
│   │   └── eda.py               # 探索性数据分析
│   ├── rf/                      # 路线一：TF-IDF + 随机森林
│   ├── fasttext/                # 路线二：FastText（词粒度 + 字粒度）
│   ├── bert/                    # 路线三：BERT-base-chinese 微调
│   ├── prune/                   # 路线四：BERT 模型剪枝
│   ├── distill/                 # 路线五：知识蒸馏（骨架）
│   ├── llm/                     # 路线六：LLM Prompt 工程（DeepSeek）
│   └── serving/                 # 统一 Web 服务入口
├── tests/                       # 测试用例
└── tutorial/                    # 教学文档 & 面经
```

## 技术路线

| 路线 | 方案 | 核心思路 |
|------|------|----------|
| 一 | TF-IDF + 随机森林 | 手工特征 + 传统集成学习 |
| 二 | FastText | 子词嵌入，分字粒度和词粒度两种 |
| 三 | BERT 微调 | 预训练大模型下游微调 |
| 四 | BERT 剪枝 | L1 非结构化剪枝，压缩 30% 参数 |
| 五 | 知识蒸馏 | BERT → 小模型蒸馏（待实现） |
| 六 | LLM Prompt 工程 | DeepSeek API 零样本分类 |

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key（仅路线六需要）

```bash
cp .env.example .env
# 编辑 .env 文件，填入 DeepSeek API Key
```

### 3. 数据预处理

```bash
python -m src.common.data_utils
```

### 4. 训练 & 预测

```bash
# 随机森林
python -m src.rf.data_process    # 数据预处理
python -m src.rf.train           # 训练
python -m src.rf.test            # 测试
python -m src.rf.predict         # 单条预测

# FastText
python -m src.fasttext.train_char_default  # 字粒度默认参数
python -m src.fasttext.train_word_auto     # 词粒度自动调参

# BERT 微调
python -m src.bert.train          # 训练 + 测试
python -m src.bert.quantization   # 动态量化
python -m src.prune.prune         # 模型剪枝

# LLM 零样本
python -m src.llm.predict         # 单条预测
python -m src.llm.prompt_classify # 批量分类
```

### 5. 启动 Web 服务

每条路线均提供三种服务方式，以随机森林为例：

```bash
# Flask
python -m src.rf.flask_server

# FastAPI (带 Swagger 文档)
python -m src.rf.fastapi_server

# Streamlit (交互式界面)
streamlit run src/rf/streamlit_app.py
```

其他路线替换 `rf` 为对应模块名即可（`fasttext` / `bert` / `prune` / `llm`）。

## 数据集

- **来源**: 中文新闻短文本
- **类别数**: 10 类 — `finance / realty / stocks / education / science / society / politics / sports / game / entertainment`
- **格式**: TSV，每行 `文本\t类别索引`

## 核心依赖

- **传统 ML**: scikit-learn, jieba
- **FastText**: fasttext
- **深度学习**: PyTorch, transformers
- **LLM API**: openai (DeepSeek 兼容)
- **Web 服务**: Flask, FastAPI, Streamlit

## License

MIT

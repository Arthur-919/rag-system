# 🧠 RAG 智能知识库问答系统

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![RAGAS](https://img.shields.io/badge/RAGAS-Evaluated-green.svg)](https://docs.ragas.io/)
[![Ollama](https://img.shields.io/badge/Ollama-0.1+-orange.svg)](https://ollama.ai/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-red.svg)](https://flask.palletsprojects.com/)

基于 **Ollama + ChromaDB + 混合检索 + Reranker** 的本地化 RAG 问答系统，支持多格式文档检索，针对中文知识库深度优化。

---

## 📊 评估结果

经过 RAGas 框架系统评估，当前版本达到**生产可用**水平：

| 指标                          | 分数       | 状态       |
| ----------------------------- | ---------- | ---------- |
| 忠实度 (Faithfulness)         | **0.9024** | ✅ 优秀     |
| 答案相关性 (AnswerRelevancy)  | 0.7901     | ⚠️ 良好     |
| 上下文精度 (ContextPrecision) | **0.9000** | ✅ 优秀     |
| 上下文召回 (ContextRecall)    | **0.8933** | ✅ 优秀     |
| **综合均分**                  | **0.8715** | ✅ 生产可用 |

### 优化前后对比
优化前 										优化后
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
忠实度 0.65 ████████████████░░░░░░░░  0.90 ████████████████████████████████
相关性 0.89 ██████████████████████████ 0.79 ██████████████████████████░░░
精度 0.69 ██████████████████░░░░░░ 0.90 ████████████████████████████████
召回率 0.48 ████████████░░░░░░░░░░░░ 0.89 ████████████████████████████████
综合分 0.68 ██████████████████░░░░░░ 0.87 ██████████████████████████████░
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

text

> 📈 **核心成果**：召回率提升 **85%**，综合性能提升 **28%**

---

## ✨ 核心特性

| 特性             | 说明                                             |
| ---------------- | ------------------------------------------------ |
| 📄 **多格式支持** | TXT、Markdown、DOCX、PDF                         |
| 🔍 **混合检索**   | BM25（权重0.6）+ 向量检索（权重0.4），关键词优先 |
| 📚 **父子块分词** | 子块400字精确匹配 + 父块2500字完整上下文         |
| 🎯 **Reranker**   | BAAI/bge-reranker-v2-m3 重排序，提升精度         |
| 💾 **智能缓存**   | LRU缓存 + 文档哈希增量更新                       |
| 🌊 **流式响应**   | SSE实时输出，用户体验友好                        |
| 🔧 **本地部署**   | 基于Ollama，无需GPU，数据不出域                  |
| 📊 **调试接口**   | 检索质量分析，支持RAGas评估                      |

---

## 🏗️ 系统架构
用户问题
│
▼
┌─────────────────────────────────────────────────────┐
│ Step 1: 查询改写 │
│ 使用LLM消除指代歧义（"它"、"这个" → 明确实体） │
└─────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────┐
│ Step 2: 混合检索（子块索引） │
│ ┌─────────────────┐ ┌─────────────────┐ │
│ │ BM25 检索 │ + │ 向量检索 │ │
│ │ (jieba分词) │ │ (BGE嵌入) │ │
│ │ 权重: 0.6 │ │ 权重: 0.4 │ │
│ └─────────────────┘ └─────────────────┘ │
│ ↓ ↓ │
│ ┌─────────────────────────────────┐ │
│ │ Ensemble 融合（Top-12） │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────┐
│ Step 3: 父子块映射 │
│ 子块ID → 父块ID → 返回完整上下文（2500字） │
└─────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────┐
│ Step 4: Reranker 重排序 │
│ CrossEncoder (bge-reranker-v2-m3) 精准打分 │
│ 保留 Top-8 最相关父块 │
└─────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────┐
│ Step 5: LLM 生成 │
│ 模型: Qwen2.5-7B | Temperature: 0.15 │
│ 严格遵循参考资料 + 来源标注 │
└─────────────────────────────────────────────────────┘
│
▼
最终答案 + 📚 [来源: doc1.pdf, doc2.md]

text

---

## 🚀 快速开始

### 前置要求

```bash
# 1. 安装 Ollama (Linux/macOS)
curl -fsSL https://ollama.com/install.sh | sh

# Windows 用户请访问 https://ollama.com/download 下载安装包

# 2. 拉取所需模型
ollama pull qwen2.5:7b                      # LLM 模型（约4.7GB）
ollama pull quentinz/bge-large-zh-v1.5      # 嵌入模型（约1.3GB）
安装与运行
bash
# 1. 克隆项目
git clone https://github.com/yourname/rag-system.git
cd rag-system

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 准备文档
mkdir docs
cp your_documents.pdf docs/
# 支持格式: .txt, .md, .docx, .pdf

# 5. 启动服务
python app.py
访问 http://localhost:8080 开始使用

⚙️ 配置说明
核心配置在 app.py 的 CONFIG 字典中：

python
CONFIG = {
    # ========== 模型配置 ==========
    "ollama_url": "http://127.0.0.1:11434",
    "llm_model": "qwen2.5:7b",
    "embedding_model": "quentinz/bge-large-zh-v1.5",
    
    # ========== 分词策略（关键优化点）==========
    "parent_chunk_size": 2500,     # 父块：完整上下文
    "parent_chunk_overlap": 200,
    "child_chunk_size": 400,       # 子块：精确检索
    "child_chunk_overlap": 80,
    
    # ========== 检索策略 ==========
    "retrieval_k": 12,             # 初选候选数
    "bm25_weight": 0.6,            # BM25 权重（关键词优先）
    "vector_weight": 0.4,
    "reranker_top_k": 8,           # 重排序后保留数
    
    # ========== 生成策略 ==========
    "temperature": 0.15,           # 低温度减少幻觉
    "enable_query_rewriting": True,
    
    # ========== 性能优化 ==========
    "embed_batch_size": 50,        # 批量嵌入大小
    "cache_size": 128,             # LRU缓存大小
}
参数调优建议
场景	chunk_size	BM25权重	temperature	说明
通用知识库	400/2500	0.5	0.15	默认配置
技术文档（关键词多）	300/2000	0.7	0.10	提高BM25权重
长文本/小说	500/3000	0.3	0.20	提高向量权重
事实问答（低幻觉）	400/2500	0.5	0.05	极低温度
📁 项目结构
text
rag-system/
├── app.py                  # 主程序（Flask API + RAG逻辑）
├── requirements.txt        # Python依赖
├── docs/                   # 📄 文档目录（用户上传）
│   ├── example.pdf
│   └── example.md
├── chroma_db/              # 💾 向量数据库（自动生成）
│   ├── chroma.sqlite3
│   └── ...
├── templates/
│   └── index.html          # 🎨 Web对话界面
└── README.md
🔧 API 接口
1. 流式问答（推荐）
bash
POST /ask_stream
Content-Type: application/json

{
    "question": "什么是支持向量机？"
}
响应（Server-Sent Events）：

text
data: {"token": "支持向量机"}
data: {"token": "（SVM）是一种"}
...
data: [DONE]
2. 同步问答
bash
POST /ask
Content-Type: application/json

{
    "question": "逻辑回归和线性回归的区别"
}

# 响应
{
    "answer": "逻辑回归用于分类问题..."
}
3. 调试接口（分析检索质量）
bash
POST /debug/retrieve
Content-Type: application/json

{
    "question": "KMeans算法的原理"
}

# 返回子块和父块的检索结果，不经过LLM
4. 健康检查
bash
GET /health

# 响应
{
    "status": "ok",
    "retriever_ready": true
}
📈 优化历程
版本	召回率	忠实度	综合分	关键改进
v1.0	0.48	0.65	0.68	基础向量检索
v2.0	0.75	0.78	0.77	+ 混合检索(BM25)
v3.0	0.85	0.85	0.83	+ 父子块分词
v4.0	0.89	0.90	0.87	+ Reranker + 查询改写
关键发现
BM25权重 > 向量权重：中文技术文档场景下，关键词匹配比语义相似更重要

父子块分词贡献最大：召回率提升85%，是性价比最高的优化

低温度有效控制幻觉：temperature=0.15 配合严格prompt，忠实度达90%

Reranker精度提升有限：主要作用于排名前几位的文档，边际收益约3-5%

📦 依赖清单
text
# 核心框架
langchain>=0.1.0
langchain-chroma>=0.1.0
langchain-community>=0.0.10
langchain-classic>=0.0.10

# Web框架
flask>=2.3.0

# 检索与嵌入
sentence-transformers>=2.2.0  # Reranker
chromadb>=0.4.0

# 中文处理
jieba>=0.42.1

# 文档解析
pypdf>=3.0.0
docx2txt>=0.8
unstructured>=0.10.0

# 工具
requests>=2.31.0
安装命令：

bash
pip install langchain langchain-chroma langchain-community flask jieba pypdf docx2txt unstructured sentence-transformers chromadb requests
🐛 常见问题
Q1: 启动时提示 "Connection refused"？
A: 确保 Ollama 服务已启动：

bash
ollama serve  # 后台运行
ollama list   # 确认模型已拉取
Q2: 中文检索效果不好？
A: 调整 BM25 权重：

python
"bm25_weight": 0.7,  # 提高关键词权重
Q3: Windows 下出现文件锁错误？
A: 程序已实现重试机制，会自动等待5次。如持续失败，手动删除 chroma_db 目录后重启。

Q4: 嵌入速度很慢？
A: 调整 batch_size：

python
"embed_batch_size": 100,  # 根据GPU内存调整
Q5: 生成的答案有幻觉？
A:

降低 temperature 至 0.05

检查召回率是否过低（< 0.7）

优化文档质量，确保信息完整

🗺️ 路线图
Docker 容器化部署（含 docker-compose）

支持更多文档格式（Excel、PPT、HTML）

多轮对话上下文支持（ConversationBufferMemory）

检索结果的用户反馈机制（RLHF）

接入更多向量数据库（Milvus、Qdrant）

添加单元测试（pytest）

API 限流与鉴权

监控告警（Prometheus + Grafana）

🤝 贡献
欢迎提交 Issue 和 Pull Request

Fork 本项目

创建特性分支 (git checkout -b feature/AmazingFeature)

提交更改 (git commit -m 'Add some AmazingFeature')

推送到分支 (git push origin feature/AmazingFeature)

开启 Pull Request

📄 License
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
...

👨‍💻 作者
Arthur-919 - [GitHub](https://github.com/Arthur-919)


🙏 致谢
RAGAS - RAG 评估框架

Ollama - 本地 LLM 部署

BAAI/bge-reranker - 重排序模型

LangChain - LLM 应用框架

Chroma - 向量数据库
```
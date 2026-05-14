import sys
import os
import hashlib
import json
import time

# 修复 Windows GBK 终端无法打印 Unicode 符号的问题
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENDPOINT'] = 'https://hf-mirror.com'  # sentence_transformers / huggingface_hub 新版用

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
import jieba
import requests
from typing import List
import shutil
from pathlib import Path
from sentence_transformers import CrossEncoder

app = Flask(__name__)

# ========== 配置 ==========
CONFIG = {
    "ollama_url": "http://127.0.0.1:11434",
    "llm_model": "qwen2.5:7b",
    "embedding_model": "quentinz/bge-large-zh-v1.5",  # 中文专用嵌入模型，1024维
    "docs_dir": "./docs",

    # 父子块分词配置：父块保留完整上下文，子块用于精确检索
    # bge-large-zh-v1.5 限 512 tokens，子块需控制在 ~400 字以内
    "parent_chunk_size": 2500,
    "parent_chunk_overlap": 200,
    "child_chunk_size": 400,
    "child_chunk_overlap": 80,
    "retrieval_k": 12,  # 候选池（平衡速度与召回）
    "bm25_weight": 0.6,  # BM25 在中文关键词匹配上更精准
    "vector_weight": 0.4,
    "embed_batch_size": 50,
    "cache_size": 128,
    "reranker_model": "BAAI/bge-reranker-v2-m3",
    "reranker_top_k": 8,  # 重排序后保留数量
    "enable_query_rewriting": True,  # 是否启用查询改写（处理模糊指代）
    "system_prompt": """你是严格基于机器学习资料库的问答助手。

知识库特点：
- 涵盖机器学习全链路知识：数学基础、数据处理、特征工程、经典算法、模型评估、优化策略、深度学习、工程落地等
- 内容以自然段落形式组织，每个知识点都有完整描述
- 算法包含：线性回归、逻辑回归、KNN、朴素贝叶斯、决策树、集成学习（随机森林/XGBoost/LightGBM）、SVM、KMeans、DBSCAN、PCA等
- 每个算法都包含：原理、损失函数、求解方式、优缺点、适用场景

核心原则（严格遵守）：
1. 你只能用【参考资料】中的信息回答问题
2. 每一条事实性陈述都必须标注来源[片段X]，没有来源的事实不得输出
3. 如果【参考资料】只能部分回答问题，先输出资料中的内容（标注[片段X]），然后单独一行明确标注「⚠️ 以下为补充知识：」，再补充你的知识
4. 如果【参考资料】完全无法回答，直接回复「参考资料中未涵盖此问题」——禁止编造、禁止猜测、禁止用一般常识替代
5. 遇到资料矛盾时，列出矛盾点让用户自行判断

回答格式：
- 如果问到具体算法，包含：原理 + 核心公式/参数 + 优缺点
- 如果问到对比类问题，使用表格或对比列表
- 回答控制在200-500字之间，除非问题要求详细展开""",
}

# 全局变量
retriever = None
vector_store = None
_session = None
_reranker = None


def get_session():
    """复用 HTTP Session，避免每次请求都建连"""
    global _session
    if _session is None:
        _session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=1,
        )
        _session.mount('http://', adapter)
        _session.mount('https://', adapter)
    return _session


# ========== QA 缓存 ==========
class LRUCache:
    def __init__(self, maxsize=128):
        self._cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()


qa_cache = LRUCache(maxsize=CONFIG["cache_size"])


# ========== 1. jieba 分词 ==========
def jieba_tokenizer(text):
    return list(jieba.cut_for_search(text))


# ========== 1.5 带分数的向量检索器 ==========
class ScoredVectorRetriever(BaseRetriever):
    """封装 similarity_search_with_score，将距离分数存入 doc.metadata"""
    vector_store: object = None
    k: int = 10

    def _get_relevant_documents(self, query: str):
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=self.k)
        docs = []
        for doc, score in docs_with_scores:
            doc.metadata['vector_score'] = float(score)
            docs.append(doc)
        return docs


# ========== 1.6 父子块检索器（含 Reranker）==========
class ParentChildRetriever:
    """父子块检索器：用子块做索引和匹配，检索时返回父块，并用 Reranker 重排序"""
    def __init__(self, child_retriever, parent_chunks, reranker=None, reranker_top_k=5):
        self.child_retriever = child_retriever
        self.parent_chunks = parent_chunks
        self.reranker = reranker
        self.reranker_top_k = reranker_top_k

    def invoke(self, query):
        child_docs = self.child_retriever.invoke(query)
        seen = set()
        parent_docs = []
        for doc in child_docs:
            pid = doc.metadata.get('parent_idx')
            if pid is not None and pid not in seen:
                seen.add(pid)
                parent_docs.append(self.parent_chunks[pid])

        # Reranker 重排序：用 CrossEncoder 对父块打分，取 top_k
        if self.reranker and len(parent_docs) > self.reranker_top_k:
            pairs = [[query, doc.page_content] for doc in parent_docs]
            scores = self.reranker.predict(pairs)
            sorted_docs = sorted(zip(parent_docs, scores), key=lambda x: x[1], reverse=True)
            parent_docs = [doc for doc, _ in sorted_docs[:self.reranker_top_k]]

        return parent_docs


# ========== 2. 自定义 Embedding（批量 + Session 复用）==========
class DirectOllamaEmbeddings(Embeddings):
    def __init__(self, model="bge-m3", base_url="http://127.0.0.1:11434", batch_size=50):
        self.model = model
        self.base_url = base_url
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        total = len(texts)
        session = get_session()
        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            print(f"  Embedding: {min(i + self.batch_size, total)}/{total}")
            response = session.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": batch},
                timeout=300
            )
            if response.status_code != 200:
                raise Exception(f"Embedding 失败: {response.text}")
            embeddings.extend(response.json()["embeddings"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = get_session().post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": [text]},
            timeout=120
        )
        if response.status_code != 200:
            raise Exception(f"Embedding 失败: {response.text}")
        return response.json()["embeddings"][0]


# ========== 3. 文档哈希 ==========
def get_docs_hash():
    hasher = hashlib.md5()
    docs_path = Path(CONFIG["docs_dir"])
    if not docs_path.exists():
        return ""
    # 将分词参数和 embedding 模型纳入哈希，参数变更时自动触发重建
    chunk_config = f"{CONFIG['embedding_model']}_{CONFIG['parent_chunk_size']}_{CONFIG['parent_chunk_overlap']}_{CONFIG['child_chunk_size']}_{CONFIG['child_chunk_overlap']}"
    hasher.update(chunk_config.encode())
    for f in sorted(docs_path.iterdir()):
        if f.suffix.lower() in {'.txt', '.md', '.docx', '.pdf'}:
            hasher.update(f.read_bytes())
            hasher.update(f.name.encode())
    return hasher.hexdigest()


# ========== 4. 加载文档（并行）==========
def load_documents():
    docs_path = Path(CONFIG["docs_dir"])
    if not docs_path.exists():
        os.makedirs(docs_path)
        return []

    loaders_map = {
        ".txt": lambda p: TextLoader(str(p), encoding='utf-8'),
        ".md": lambda p: UnstructuredMarkdownLoader(str(p), mode="elements"),
        ".docx": lambda p: Docx2txtLoader(str(p)),
        ".pdf": lambda p: PyPDFLoader(str(p)),
    }

    files = [f for f in docs_path.iterdir() if f.suffix.lower() in loaders_map]

    def load_one(file_path):
        try:
            loader = loaders_map[file_path.suffix.lower()](file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = file_path.name
            print(f"✅ 加载: {file_path.name}")
            return docs
        except Exception as e:
            print(f"❌ 失败 {file_path.name}: {e}")
            return []

    documents = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(load_one, f): f for f in files}
        for future in as_completed(futures):
            documents.extend(future.result())

    return documents


# ========== 5. 父子块分词 ==========
def split_into_parent_child(documents):
    """将文档先切分为父块（大块），再将每个父块切分为子块（小块）"""
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["parent_chunk_size"],
        chunk_overlap=CONFIG["parent_chunk_overlap"],
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    parent_chunks = parent_splitter.split_documents(documents)
    print(f"📄 {len(documents)} 个文档 → {len(parent_chunks)} 个父块")

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["child_chunk_size"],
        chunk_overlap=CONFIG["child_chunk_overlap"],
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )

    child_chunks = []
    for parent_idx, parent_chunk in enumerate(parent_chunks):
        sub_chunks = child_splitter.split_documents([parent_chunk])
        for sub_chunk in sub_chunks:
            sub_chunk.metadata['parent_idx'] = parent_idx
            child_chunks.append(sub_chunk)

    print(f"  → {len(child_chunks)} 个子块（用于索引）")
    return parent_chunks, child_chunks


def _remove_chroma_db(db_dir):
    """安全删除 ChromaDB 目录（处理 Windows 文件锁问题）"""
    import time
    for attempt in range(5):
        try:
            shutil.rmtree(db_dir)
            return
        except PermissionError:
            if attempt < 4:
                print(f"  ⚠️ 文件被占用，等待重试 ({attempt + 1}/5)...")
                time.sleep(1.5)
            else:
                print("  ⚠️ 部分文件无法删除，尝试逐文件清理...")
                for root, dirs, files in os.walk(db_dir, topdown=False):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except PermissionError:
                            pass
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except OSError:
                            pass


# ========== 6. 初始化检索器（增量复用 Chroma DB）==========
def init_retriever(force=False):
    global retriever, vector_store

    print("=" * 50)
    print("正在初始化 RAG 系统（父子块分词模式）...")
    print(f"  父块大小: {CONFIG['parent_chunk_size']}, 重叠: {CONFIG['parent_chunk_overlap']}")
    print(f"  子块大小: {CONFIG['child_chunk_size']}, 重叠: {CONFIG['child_chunk_overlap']}")
    print("=" * 50)

    hash_file = Path("./chroma_db/docs_hash.txt")
    current_hash = get_docs_hash()
    chroma_exists = Path("./chroma_db").exists() and any(Path("./chroma_db").iterdir())

    # 尝试复用已有向量库
    if not force and chroma_exists and hash_file.exists():
        cached_hash = hash_file.read_text().strip()
        if cached_hash == current_hash and cached_hash:
            print("✅ 文档和分词参数未变更，复用已有向量库")
            try:
                embeddings = DirectOllamaEmbeddings(
                    model=CONFIG["embedding_model"],
                    base_url=CONFIG["ollama_url"],
                    batch_size=CONFIG["embed_batch_size"],
                )
                vector_store = Chroma(
                    persist_directory="./chroma_db",
                    embedding_function=embeddings,
                )
                vector_retriever = ScoredVectorRetriever(
                    vector_store=vector_store, k=CONFIG["retrieval_k"]
                )

                # 重新加载文档并做父子块分词（BM25 不支持持久化，必须重建）
                documents = load_documents()
                if not documents:
                    print("⚠️ 请将文档放入 docs 文件夹")
                    return False

                parent_chunks, child_chunks = split_into_parent_child(documents)

                bm25_retriever = BM25Retriever.from_documents(
                    documents=child_chunks,
                    preprocess_func=jieba_tokenizer
                )
                bm25_retriever.k = CONFIG["retrieval_k"]

                ensemble = EnsembleRetriever(
                    retrievers=[bm25_retriever, vector_retriever],
                    weights=[CONFIG["bm25_weight"], CONFIG["vector_weight"]]
                )

                reranker = load_reranker()
                retriever = ParentChildRetriever(
                    ensemble, parent_chunks,
                    reranker=reranker,
                    reranker_top_k=CONFIG["reranker_top_k"]
                )
                qa_cache.clear()
                print("✅ RAG 系统就绪！（复用缓存）")
                print("=" * 50)
                return True
            except Exception as e:
                print(f"⚠️ 缓存加载失败，将完整重建: {e}")
                # 关闭可能已打开的 ChromaDB 连接，释放文件锁
                try:
                    vector_store._client.close()
                except Exception:
                    pass

    # === 完整重建 ===
    documents = load_documents()
    if not documents:
        print("⚠️ 请将文档放入 docs 文件夹")
        return False

    parent_chunks, child_chunks = split_into_parent_child(documents)

    # 向量库（用子块建索引）
    print("正在加载 Embedding 模型...")
    embeddings = DirectOllamaEmbeddings(
        model=CONFIG["embedding_model"],
        base_url=CONFIG["ollama_url"],
        batch_size=CONFIG["embed_batch_size"],
    )

    print("正在创建向量数据库（子块索引）...")
    # 关闭旧连接，释放文件锁
    if vector_store is not None:
        try:
            vector_store._client.close()
        except Exception:
            pass
    if os.path.exists("./chroma_db"):
        _remove_chroma_db("./chroma_db")

    t0 = time.time()
    vector_store = Chroma.from_documents(
        documents=child_chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
    )
    print(f"✅ 向量数据库创建完成（{time.time() - t0:.1f}s）")

    hash_file.write_text(current_hash)

    vector_retriever = ScoredVectorRetriever(
        vector_store=vector_store, k=CONFIG["retrieval_k"]
    )

    # BM25（用子块建索引）
    print("正在构建 BM25 索引（子块）...")
    t0 = time.time()
    bm25_retriever = BM25Retriever.from_documents(
        documents=child_chunks,
        preprocess_func=jieba_tokenizer
    )
    bm25_retriever.k = CONFIG["retrieval_k"]
    print(f"✅ BM25 索引构建完成（{time.time() - t0:.1f}s）")

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[CONFIG["bm25_weight"], CONFIG["vector_weight"]]
    )

    # 用父子块检索器包装：子块匹配 → 返回父块，Reranker 重排序
    reranker = load_reranker()
    retriever = ParentChildRetriever(
        ensemble, parent_chunks,
        reranker=reranker,
        reranker_top_k=CONFIG["reranker_top_k"]
    )
    print("✅ 父子块混合检索器已创建（含 Reranker）")

    qa_cache.clear()
    print("✅ RAG 系统就绪！")
    print("=" * 50)
    return True


# ========== 6.5 加载 Reranker ==========
def load_reranker():
    global _reranker
    if _reranker is None:
        print(f"正在加载 Reranker 模型: {CONFIG['reranker_model']} ...")
        _reranker = CrossEncoder(CONFIG['reranker_model'])
        print("✅ Reranker 模型加载完成")
    return _reranker


# ========== 6.6 查询改写 ==========
def rewrite_query(original_query):
    """用 LLM 轻量改写模糊查询，消除指代词（如'它''这个'）"""
    if not CONFIG.get("enable_query_rewriting", True):
        return original_query

    rewrite_prompt = f"""你的任务是将模糊的后续问题改写为独立、明确的检索查询。

规则：
- 如果问题包含"它"、"他"、"这个"、"那个"、"其"等指代词，根据语义补全为明确表述
- 如果问题已经是完整独立的，原样返回
- 只返回改写后的问题，不要加任何解释

原始问题：{original_query}
改写后的问题："""

    try:
        response = get_session().post(
            f"{CONFIG['ollama_url']}/api/generate",
            json={
                "model": CONFIG["llm_model"],
                "prompt": rewrite_prompt,
                "temperature": 0.1,
                "stream": False
            },
            timeout=30
        )
        if response.status_code == 200:
            rewritten = response.json()["response"].strip()
            if rewritten and rewritten != original_query:
                print(f"🔄 查询改写: \"{original_query}\" → \"{rewritten}\"")
                return rewritten
        return original_query
    except Exception as e:
        print(f"⚠️ 查询改写失败，使用原查询: {e}")
        return original_query


# ========== 7. 构建 prompt ==========
def build_prompt(question, docs):
    top_docs = docs[:CONFIG["retrieval_k"]]
    parts = []
    for i, doc in enumerate(top_docs, 1):
        source = doc.metadata.get('source', '未知')
        parts.append(f"[片段{i}] 来源: {source}\n{doc.page_content}")
    context = "\n\n---\n\n".join(parts)

    return f"""【参考资料】
{context}

【用户问题】
{question}"""


# ========== 8. 检索 ==========
def retrieve(question):
    global retriever
    if retriever is None:
        return None, None, None

    # 查询改写：消除模糊指代，提升召回率
    search_query = rewrite_query(question)

    docs = retriever.invoke(search_query)
    if not docs:
        return None, None, None

    sources = list(set([doc.metadata.get('source', '未知') for doc in docs]))
    # 构建 prompt 时仍使用用户原始问题，保持对话自然
    prompt = build_prompt(question, docs)
    return docs, prompt, sources


# ========== 9. 普通问答（带缓存）==========
def ask(question):
    # 缓存命中
    cached = qa_cache.get(question)
    if cached:
        return cached

    if retriever is None:
        if not init_retriever():
            return "知识库未就绪，请先添加文档"

    try:
        docs, prompt, sources = retrieve(question)
        if docs is None:
            return "未找到相关内容"

        t0 = time.time()
        response = get_session().post(
            f"{CONFIG['ollama_url']}/api/generate",
            json={
                "model": CONFIG["llm_model"],
                "system": CONFIG["system_prompt"],
                "prompt": prompt,
                "temperature": 0.15,
                "stream": False
            },
            timeout=120
        )

        if response.status_code != 200:
            return f"LLM 调用失败: {response.text}"

        answer = response.json()["response"]
        print(f"LLM 生成耗时: {time.time() - t0:.1f}s")

        if sources:
            answer += f"\n\n📚 [来源: {', '.join(sources)}]"

        qa_cache.put(question, answer)
        return answer

    except Exception as e:
        print(f"查询错误: {e}")
        return f"查询失败：{e}"


# ========== 10. Flask 路由 ==========
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def handle_ask():
    data = request.json
    question = data.get('question', '')
    answer = ask(question)
    return jsonify({'answer': answer})


@app.route('/ask_stream', methods=['POST'])
def handle_ask_stream():
    data = request.json
    question = data.get('question', '')

    def generate():
        # 先检查是否需要初始化
        if retriever is None:
            if not init_retriever():
                yield f"data: {json.dumps({'error': '知识库未就绪，请先添加文档'})}\n\n"
                return

        # 缓存命中
        cached = qa_cache.get(question)
        if cached:
            yield f"data: {json.dumps({'token': cached})}\n\n"
            yield "data: [DONE]\n\n"
            return

        try:
            docs, prompt, sources = retrieve(question)
            if docs is None:
                yield f"data: {json.dumps({'token': '未找到相关内容'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            session = get_session()
            response = session.post(
                f"{CONFIG['ollama_url']}/api/generate",
                json={
                    "model": CONFIG["llm_model"],
                    "system": CONFIG["system_prompt"],
                    "prompt": prompt,
                    "temperature": 0.15,
                    "stream": True
                },
                timeout=300,
                stream=True
            )

            full_answer = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode())
                        token = chunk.get("response", "")
                        if token:
                            full_answer += token
                            yield f"data: {json.dumps({'token': token})}\n\n"
                    except json.JSONDecodeError:
                        continue

            # 附加来源
            if sources:
                source_text = f"\n\n📚 [来源: {', '.join(sources)}]"
                full_answer += source_text
                yield f"data: {json.dumps({'token': source_text})}\n\n"

            qa_cache.put(question, full_answer)

        except Exception as e:
            print(f"流式查询错误: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        }
    )


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'retriever_ready': retriever is not None,
        'rag_ready': retriever is not None
    })


@app.route('/api/documents', methods=['GET'])
def get_documents():
    docs_path = Path(CONFIG["docs_dir"])
    if not docs_path.exists():
        return jsonify({"count": 0, "documents": []})

    documents = []
    for ext in ['*.txt', '*.md', '*.docx', '*.pdf']:
        for file_path in docs_path.glob(ext):
            documents.append({"name": file_path.name})

    return jsonify({"count": len(documents), "documents": documents})


@app.route('/debug/retrieve', methods=['POST'])
def debug_retrieve():
    """调试端点：查看检索到的子块和父块内容，不经过 LLM"""
    if retriever is None:
        return jsonify({"error": "检索器未初始化"})

    question = request.json.get('question', '')
    if not question:
        return jsonify({"error": "请提供 question"})

    # 获取子块检索结果（绕过 ParentChildRetriever）
    child_docs = retriever.child_retriever.invoke(question)

    child_results = []
    for i, doc in enumerate(child_docs):
        child_results.append({
            "rank": i + 1,
            "source": doc.metadata.get('source', '未知'),
            "parent_idx": doc.metadata.get('parent_idx', 'N/A'),
            "content_preview": doc.page_content[:200] + "...",
        })

    # 获取父块结果（经过 ParentChildRetriever）
    parent_docs = retriever.invoke(question)
    parent_results = []
    for i, doc in enumerate(parent_docs):
        parent_results.append({
            "rank": i + 1,
            "source": doc.metadata.get('source', '未知'),
            "content_preview": doc.page_content[:300] + "...",
        })

    return jsonify({
        "child_hits": len(child_results),
        "parent_hits": len(parent_results),
        "child_results": child_results,
        "parent_results": parent_results,
    })


# ========== 11. 启动 ==========
if __name__ == '__main__':
    jieba.initialize()

    print("=" * 50)
    print("RAG 知识库问答系统（父子块分词版）")
    print(f"   LLM: {CONFIG['llm_model']}")
    print(f"   Embedding: {CONFIG['embedding_model']}")
    print(f"   父块大小: {CONFIG['parent_chunk_size']}, 子块大小: {CONFIG['child_chunk_size']}")
    print(f"   检索数量: {CONFIG['retrieval_k']}")
    print(f"   批量 Embed: {CONFIG['embed_batch_size']}条/次")
    print(f"   支持格式: .txt, .md, .docx, .pdf")
    print(f"   访问: http://localhost:8080")
    print("=" * 50)

    init_retriever()
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
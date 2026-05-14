"""
RAG 核心引擎 —— 检索器初始化、查询改写、检索、问答
"""

import os
import json
import time
import shutil
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

from config import CONFIG
from cache import qa_cache
from session import get_session
from embedding import DirectOllamaEmbeddings
from retrievers import ScoredVectorRetriever, ParentChildRetriever
from document_loader import load_documents, split_into_parent_child, get_docs_hash, jieba_tokenizer

# 全局状态
retriever = None
vector_store = None
_reranker = None


def _remove_chroma_db(db_dir):
    """安全删除 ChromaDB 目录（处理 Windows 文件锁问题）"""
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


def load_reranker():
    """加载 Reranker 模型（CrossEncoder），全局复用"""
    global _reranker
    if _reranker is None:
        print(f"正在加载 Reranker 模型: {CONFIG['reranker_model']} ...")
        _reranker = CrossEncoder(CONFIG['reranker_model'])
        print("✅ Reranker 模型加载完成")
    return _reranker


def _build_ensemble_and_retriever(parent_chunks, child_chunks):
    """构建混合检索器：BM25 + 向量 → Ensemble → 父子块包装 + Reranker"""
    global retriever, vector_store

    vector_retriever = ScoredVectorRetriever(
        vector_store=vector_store, k=CONFIG["retrieval_k"]
    )

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


def init_retriever(force=False):
    """初始化检索器（增量复用 Chroma DB）"""
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

                documents = load_documents()
                if not documents:
                    print("⚠️ 请将文档放入 docs 文件夹")
                    return False

                parent_chunks, child_chunks = split_into_parent_child(documents)
                _build_ensemble_and_retriever(parent_chunks, child_chunks)

                qa_cache.clear()
                print("✅ RAG 系统就绪！（复用缓存）")
                print("=" * 50)
                return True
            except Exception as e:
                print(f"⚠️ 缓存加载失败，将完整重建: {e}")
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

    print("正在加载 Embedding 模型...")
    embeddings = DirectOllamaEmbeddings(
        model=CONFIG["embedding_model"],
        base_url=CONFIG["ollama_url"],
        batch_size=CONFIG["embed_batch_size"],
    )

    print("正在创建向量数据库（子块索引）...")
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

    print("正在构建 BM25 索引（子块）...")
    t0 = time.time()
    _build_ensemble_and_retriever(parent_chunks, child_chunks)
    print(f"✅ BM25 索引构建完成（{time.time() - t0:.1f}s）")
    print("✅ 父子块混合检索器已创建（含 Reranker）")

    qa_cache.clear()
    print("✅ RAG 系统就绪！")
    print("=" * 50)
    return True


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


def build_prompt(question, docs):
    """构建发送给 LLM 的 prompt"""
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


def retrieve(question):
    """检索相关文档（含查询改写）"""
    global retriever
    if retriever is None:
        return None, None, None

    search_query = rewrite_query(question)
    docs = retriever.invoke(search_query)
    if not docs:
        return None, None, None

    sources = list(set([doc.metadata.get('source', '未知') for doc in docs]))
    prompt = build_prompt(question, docs)
    return docs, prompt, sources


def ask(question):
    """非流式问答（带缓存）"""
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


def ask_stream_tokens(question):
    """流式问答生成器 —— 逐 token yield，供 Flask SSE 路由调用"""
    if retriever is None:
        if not init_retriever():
            raise RuntimeError("知识库未就绪，请先添加文档")

    cached = qa_cache.get(question)
    if cached:
        yield cached
        return

    docs, prompt, sources = retrieve(question)
    if docs is None:
        raise RuntimeError("未找到相关内容")

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
                    yield token
            except json.JSONDecodeError:
                continue

    if sources:
        source_text = f"\n\n📚 [来源: {', '.join(sources)}]"
        full_answer += source_text
        yield source_text

    qa_cache.put(question, full_answer)

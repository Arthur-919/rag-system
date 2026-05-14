"""
RAG 知识库问答系统 —— Flask 路由入口
"""

import sys
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import jieba

from config import CONFIG
from cache import qa_cache
from session import get_session
from rag_engine import (
    retriever, init_retriever, ask, retrieve,
    rewrite_query, build_prompt, ask_stream_tokens
)

app = Flask(__name__)


# ========== Flask 路由 ==========

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
        try:
            for token in ask_stream_tokens(question):
                yield f"data: {json.dumps({'token': token})}\n\n"
        except RuntimeError as e:
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

    child_docs = retriever.child_retriever.invoke(question)

    child_results = []
    for i, doc in enumerate(child_docs):
        child_results.append({
            "rank": i + 1,
            "source": doc.metadata.get('source', '未知'),
            "parent_idx": doc.metadata.get('parent_idx', 'N/A'),
            "content_preview": doc.page_content[:200] + "...",
        })

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


# ========== 启动 ==========

if __name__ == '__main__':
    # 修复 Windows GBK 终端无法打印 Unicode 符号的问题
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

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

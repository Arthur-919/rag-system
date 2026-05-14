"""
文档加载与父子块分词
"""

import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import jieba
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CONFIG


def jieba_tokenizer(text):
    return list(jieba.cut_for_search(text))


def get_docs_hash():
    """计算文档和分词参数的联合哈希，用于判断是否需要重建向量库"""
    hasher = hashlib.md5()
    docs_path = Path(CONFIG["docs_dir"])
    if not docs_path.exists():
        return ""
    chunk_config = f"{CONFIG['embedding_model']}_{CONFIG['parent_chunk_size']}_{CONFIG['parent_chunk_overlap']}_{CONFIG['child_chunk_size']}_{CONFIG['child_chunk_overlap']}"
    hasher.update(chunk_config.encode())
    for f in sorted(docs_path.iterdir()):
        if f.suffix.lower() in {'.txt', '.md', '.docx', '.pdf'}:
            hasher.update(f.read_bytes())
            hasher.update(f.name.encode())
    return hasher.hexdigest()


def load_documents():
    """并行加载 docs 目录中的所有文档"""
    docs_path = Path(CONFIG["docs_dir"])
    if not docs_path.exists():
        import os
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

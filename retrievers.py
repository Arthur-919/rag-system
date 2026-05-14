"""
自定义检索器 —— 带分数的向量检索器 + 父子块检索器（含 Reranker）
"""

from langchain_core.retrievers import BaseRetriever


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

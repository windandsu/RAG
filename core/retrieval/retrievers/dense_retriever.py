from .. import BaseRetriever
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class DenseRetriever(BaseRetriever):
    """基于密集向量的检索器"""

    def initialize(self):
        self.model_name = self.config.get('model_name', 'all-MiniLM-L6-v2')
        self.model = SentenceTransformer(self.model_name)
        self.document_embeddings = []
        self.documents = []

    def build_index(self, documents: List[Dict[str, Any]]):
        """构建向量索引"""
        self.documents = documents
        self.document_embeddings = self.model.encode(
            [doc.get('content', '') for doc in documents],
            show_progress_bar=True
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """执行密集向量检索"""
        # if not self.document_embeddings:
        #     return []
        if len(self.document_embeddings) == 0:
            return []
            
        query_embedding = self.model.encode(query)
        
        # 计算余弦相似度
        similarities = cosine_similarity([query_embedding], self.document_embeddings)[0]
        
        # 获取前k个相似的文档
        top_indices = np.argsort(-similarities)[:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] <= 0:  # 忽略相似度为0的文档
                continue
                
            doc = self.documents[idx].copy()
            doc['score'] = float(similarities[idx])
            results.append(doc)
            
        return results    
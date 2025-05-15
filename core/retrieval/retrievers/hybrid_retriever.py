from . import BaseRetriever
from typing import List, Dict, Any
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever

class HybridRetriever(BaseRetriever):
    """结合稀疏和密集检索的混合检索器"""

    def initialize(self):
        self.bm25_weight = self.config.get('bm25_weight', 0.5)
        self.dense_weight = self.config.get('dense_weight', 0.5)
        
        # 初始化子检索器
        bm25_config = {
            'index_path': self.config.get('index_path', 'bm25_index'),
            'embedding_model': 'bm25'
        }
        self.bm25_retriever = BM25Retriever(bm25_config)
        
        dense_config = {
            'index_path': self.config.get('index_path', 'dense_index'),
            'embedding_model': self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        }
        self.dense_retriever = DenseRetriever(dense_config)

    def build_index(self, documents: List[Dict[str, Any]]):
        """构建混合索引"""
        self.bm25_retriever.build_index(documents)
        self.dense_retriever.build_index(documents)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """执行混合检索"""
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k*2)  # 获取更多候选
        dense_results = self.dense_retriever.retrieve(query, top_k=top_k*2)  # 获取更多候选
        
        # 合并结果
        doc_map = {}
        
        # 处理BM25结果
        for doc in bm25_results:
            doc_id = doc.get('id', doc.get('content')[:20])  # 使用内容前20个字符作为备选ID
            doc_map[doc_id] = {
                'content': doc['content'],
                'bm25_score': doc['score'],
                'dense_score': 0,
                'metadata': {k: v for k, v in doc.items() if k not in ['content', 'score']}
            }
            
        # 处理密集检索结果
        for doc in dense_results:
            doc_id = doc.get('id', doc.get('content')[:20])
            
            if doc_id in doc_map:
                doc_map[doc_id]['dense_score'] = doc['score']
            else:
                doc_map[doc_id] = {
                    'content': doc['content'],
                    'bm25_score': 0,
                    'dense_score': doc['score'],
                    'metadata': {k: v for k, v in doc.items() if k not in ['content', 'score']}
                }
                
        # 计算混合分数
        for doc_id in doc_map:
            bm25_score = doc_map[doc_id]['bm25_score']
            dense_score = doc_map[doc_id]['dense_score']
            
            # 归一化分数
            if bm25_results:
                max_bm25_score = max(d['score'] for d in bm25_results)
                if max_bm25_score > 0:
                    bm25_score /= max_bm25_score
                    
            if dense_results:
                max_dense_score = max(d['score'] for d in dense_results)
                if max_dense_score > 0:
                    dense_score /= max_dense_score
                    
            # 计算加权分数
            doc_map[doc_id]['hybrid_score'] = (
                self.bm25_weight * bm25_score + 
                self.dense_weight * dense_score
            )
            
        # 按混合分数排序
        sorted_docs = sorted(
            doc_map.values(), 
            key=lambda x: x['hybrid_score'], 
            reverse=True
        )[:top_k]
        
        # 格式化结果
        results = []
        for doc in sorted_docs:
            result = {
                'content': doc['content'],
                'score': doc['hybrid_score'],
                **doc['metadata']
            }
            results.append(result)
            
        return results 
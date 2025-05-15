from .. import BaseGranularity
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticChunking(BaseGranularity):
    """基于语义的文本分块"""

    def initialize(self):
        self.model_name = self.config.get('model_name', 'all-MiniLM-L6-v2')
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        self.min_chunk_size = self.config.get('min_chunk_size', 200)
        self.model = SentenceTransformer(self.model_name)

    def process(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """将文档按语义分块"""
        content = document.get('content', '')
        if not content:
            return []
        
        # 简单的句子分割
        sentences = content.split('\n\n')  # 假设空行为段落分隔
        if not sentences:
            return []
            
        # 计算句子嵌入
        embeddings = self.model.encode(sentences)
        
        # 计算相邻句子之间的相似度
        similarities = []
        for i in range(len(sentences) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            similarities.append(sim)
        
        # 找出相似度低于阈值的分割点
        split_points = [0]
        current_size = 0
        
        for i, sim in enumerate(similarities):
            current_size += len(sentences[i])
            if sim < self.similarity_threshold and current_size >= self.min_chunk_size:
                split_points.append(i + 1)
                current_size = 0
                
        split_points.append(len(sentences))
        
        # 创建块
        chunks = []
        metadata = self.get_metadata(document)
        
        for i in range(len(split_points) - 1):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]
            
            chunk_content = "\n\n".join(sentences[start_idx:end_idx])
            if not chunk_content.strip():
                continue
                
            chunk = {
                'content': chunk_content,
                'start': start_idx,
                'end': end_idx,
                **metadata
            }
            chunks.append(chunk)
            
        return chunks
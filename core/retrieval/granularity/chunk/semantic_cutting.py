from .. import BaseGranularity
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging


class SemanticChunking(BaseGranularity):
    """基于语义的文本分块"""

    def initialize(self):
        self.model_name = self.config.get('model_name', 'all-MiniLM-L6-v2')
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        self.min_chunk_size = self.config.get('min_chunk_size', 100)
        self.model = SentenceTransformer(self.model_name)
        self.logger = logging.getLogger(__name__)


    def process(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """将文档按语义分块"""
        content = document.get('content', '')
        if not content:
            self.logger.info("文档内容为空")
            return []
        
        # 句子分割方法
        sentences = self._split_sentences(content)
        self.logger.info(f"分割成 {len(sentences)} 个句子")
        
        if len(sentences) < 2:
            self.logger.warning(f"句子数量过少 ({len(sentences)})，无法进行语义分割")
            # 退回到固定大小分块
            return self._fallback_to_fixed_chunking(content, document)
            
        # 计算句子嵌入
        embeddings = self.model.encode(sentences)
        
        # 计算相邻句子之间的相似度
        similarities = []
        for i in range(len(sentences) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            similarities.append(sim)
            self.logger.debug(f"句子 {i} 和 {i+1} 的相似度: {sim:.4f}")
        
        # 找出相似度低于阈值的分割点
        split_points = [0]
        current_size = 0
        
        for i, sim in enumerate(similarities):
            current_size += len(sentences[i])
            self.logger.debug(f"位置 {i}: 相似度={sim:.4f}, 当前块大小={current_size}, 阈值={self.similarity_threshold}, 最小块大小={self.min_chunk_size}")
            
            if sim < self.similarity_threshold and current_size >= self.min_chunk_size:
                split_points.append(i + 1)
                self.logger.info(f"找到分割点: {i+1}, 相似度={sim:.4f}")
                current_size = 0
                
        # 确保最后一个分割点是文档末尾
        if split_points[-1] != len(sentences):
            split_points.append(len(sentences))
            self.logger.info(f"添加末尾分割点: {len(sentences)}")
            
        self.logger.info(f"找到 {len(split_points)-1} 个分割点")
        
        # 创建块
        chunks = []
        metadata = self.get_metadata(document)
        
        for i in range(len(split_points) - 1):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]
            
            chunk_content = "".join(sentences[start_idx:end_idx])
            if not chunk_content.strip():
                continue
                
            chunk = {
                'content': chunk_content,
                'start': start_idx,
                'end': end_idx,
                **metadata
            }
            chunks.append(chunk)
            self.logger.info(f"创建块 {i+1}: 大小={len(chunk_content)}, 句子范围={start_idx}-{end_idx}")
            
        if not chunks:
            self.logger.warning("未创建任何块，返回整个文档作为一个块")
            chunks = [{
                'content': content,
                'start': 0,
                'end': len(sentences),
                **metadata
            }]
            
        return chunks
        
    def _split_sentences(self, content: str) -> List[str]:
        """将文本分割成句子"""
        # 使用更全面的句子分割正则表达式
        sentences = re.split(r'[。？！；;?!\n]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            # 如果仍然无法分割，将整个文本作为一个句子
            sentences = [content.strip()]
            
        return sentences
        
    def _fallback_to_fixed_chunking(self, content: str, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """当语义分割失败时，退回到固定大小分块"""
        self.logger.info("退回到固定大小分块")
        
        chunks = []
        metadata = self.get_metadata(document)
        start = 0
        
        while start < len(content):
            end = min(start + self.min_chunk_size, len(content))
            chunk_content = content[start:end]
            
            chunk = {
                'content': chunk_content,
                'start': start,
                'end': end,
                **metadata
            }
            chunks.append(chunk)
            
            # 设置重叠区域
            overlap = min(self.min_chunk_size // 5, end - start)  # 1/5的重叠
            start = end - overlap
            
        return chunks
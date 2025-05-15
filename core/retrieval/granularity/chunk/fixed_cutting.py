from .. import BaseGranularity
from typing import List, Dict, Any

class FixedChunking(BaseGranularity):
    """基于固定长度的文本分块"""

    def initialize(self):
        self.chunk_size = self.config.get('chunk_size', 500)
        self.overlap = self.config.get('overlap', 100)

    def process(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """将文档按固定长度分块"""
        content = document.get('content', '')
        if not content:
            return []
        
        metadata = self.get_metadata(document)
        chunks = []
        start = 0
        
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            # 如果不是最后一块且有足够的文本进行重叠
            if end < len(content) and end - start > self.overlap:
                end = start + self.chunk_size
            else:
                end = len(content)
            
            chunk = {
                'content': content[start:end],
                'start': start,
                'end': end,
                **metadata
            }
            chunks.append(chunk)
            
            if end == len(content):
                break
                
            # 设置下一个块的起始位置，考虑重叠
            overlap_start = max(0, end - self.overlap)
            start = overlap_start
            
        return chunks  
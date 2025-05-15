from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

class BaseGranularity(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initialize()

    @abstractmethod
    def initialize(self):
        """初始化粒度处理器"""
        pass

    @abstractmethod
    def process(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """将文档处理为指定粒度的片段"""
        pass

    def get_metadata(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """提取文档元数据"""
        return {
            'id': document.get('id', None),
            'title': document.get('title', ''),
            'source': document.get('source', '')
        }   
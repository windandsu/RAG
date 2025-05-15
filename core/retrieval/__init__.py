from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

class BaseRetriever(ABC):
    required_config_keys = ['index_path', 'embedding_model']

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.index = None
        self.embedding_model = None
        self.validate_config()
        self.initialize()

    @abstractmethod
    def initialize(self):
        """初始化检索器"""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索与查询相关的文档"""
        pass

    @abstractmethod
    def build_index(self, documents: List[Dict[str, Any]]):
        """构建检索索引"""
        pass  
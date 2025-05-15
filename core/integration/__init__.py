from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

class BaseIntegrator(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initialize()

    @abstractmethod
    def initialize(self):
        """初始化集成器"""
        pass

    @abstractmethod
    def integrate(self, generated_answer: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """将生成的答案与检索的文档集成"""
        pass

    def get_context_weight(self, doc: Dict[str, Any]) -> float:
        """获取文档的上下文权重"""
        return doc.get('score', 0.5)    
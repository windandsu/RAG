from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseDocumentProcessor(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initialize()

    @abstractmethod
    def initialize(self):
        """初始化处理器"""
        pass

    @abstractmethod
    def process(self, file_path: str) -> List[Dict[str, Any]]:
        """处理文档文件并返回文档列表"""
        pass
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

class BaseGranularity(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.database = self._load_database()
        self.initialize()

    def _load_database(self):
        # 这里假设你有一个数据库加载函数，根据实际情况修改
        database_path = self.config.get('database_path')
        if database_path:
            # #加载数据库
            # database = load_database(database_path)
            # return database
            pass
        return None

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
            'source': document.get('source', ''),
            'type': self._get_document_type(document)
        }  
    def _get_document_type(self, document: Dict[str, Any]) -> str:
        # 判断文档类型
        file_name = document.get('file_name', '')
        if file_name.endswith('.txt'):
            return 'text'
        elif file_name.endswith('.pdf'):
            return 'pdf'
        elif file_name.endswith('.docx'):
            return 'docx'
        elif file_name.endswith('.pptx'):
            return 'pptx'
        elif file_name.endswith('.xlsx'):
            return 'xlsx'
        return 'unknown' 
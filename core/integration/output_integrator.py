from . import BaseIntegrator
from typing import List, Dict, Any

class OutputIntegrator(BaseIntegrator):
    """在输出阶段将生成结果与检索结果集成"""

    def initialize(self):
        self.format = self.config.get('format', 
                                      "{answer}\n\nReferences:\n{references}")

    def integrate(self, generated_answer: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """将生成的答案与检索的参考文档集成"""
        references = "\n".join([
            f"[{i+1}] {doc.get('title', 'Document')}: {doc.get('content')[:100]}..." 
            for i, doc in enumerate(retrieved_docs)
        ])
        return self.format.format(answer=generated_answer, references=references) 
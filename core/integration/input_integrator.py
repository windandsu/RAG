from . import BaseIntegrator
from typing import List, Dict, Any

class InputIntegrator(BaseIntegrator):
    """在输入阶段将检索结果与查询集成"""

    def initialize(self):
        self.template = self.config.get('template', 
                                        "Context: {context}\nQuestion: {query}\nAnswer:")

    def integrate(self, generated_answer: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """将检索的文档作为上下文与查询一起输入到生成器"""
        context = "\n\n".join([doc['content'] for doc in retrieved_docs])
        return self.template.format(context=context, query=generated_answer) 
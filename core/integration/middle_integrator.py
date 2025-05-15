from . import BaseIntegrator
from typing import List, Dict, Any

class MiddleIntegrator(BaseIntegrator):
    """在生成过程中动态集成检索结果"""

    def initialize(self):
        self.weight_threshold = self.config.get('weight_threshold', 0.6)

    def integrate(self, generated_answer: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """根据相关性权重动态选择检索结果中的信息"""
        relevant_docs = []
        for doc in retrieved_docs:
            weight = self.get_context_weight(doc)
            if weight >= self.weight_threshold:
                relevant_docs.append(doc['content'])
        
        if not relevant_docs:
            return generated_answer
        
        context = "\n\n".join(relevant_docs)
        return f"Based on context: {context}\n\nAnswer: {generated_answer}"  
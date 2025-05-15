from . import BaseRetriever
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import string

class BM25Retriever(BaseRetriever):
    """基于BM25的稀疏检索器"""

    def initialize(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        self.tokenizer = self._custom_tokenize
        self.bm25 = None
        self.corpus = []
        self.doc_metadata = []

    def _custom_tokenize(self, text: str) -> List[str]:
        """自定义分词器，去除标点符号并转换为小写"""
        text = text.translate(str.maketrans('', '', string.punctuation))
        return word_tokenize(text.lower())

    def build_index(self, documents: List[Dict[str, Any]]):
        """构建BM25索引"""
        self.corpus = []
        self.doc_metadata = []
        
        for doc in documents:
            content = doc.get('content', '')
            if not content:
                continue
                
            self.corpus.append(self.tokenizer(content))
            self.doc_metadata.append({
                'id': doc.get('id'),
                'title': doc.get('title', ''),
                'source': doc.get('source', '')
            })
            
        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """执行BM25检索"""
        if self.bm25 is None:
            return []
            
        tokenized_query = self.tokenizer(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # 获取前k个文档的索引
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if doc_scores[idx] <= 0:  # 忽略得分为0的文档
                continue
                
            doc = {
                'content': " ".join(self.corpus[idx]),
                'score': doc_scores[idx],
                **self.doc_metadata[idx]
            }
            results.append(doc)
            
        return results  
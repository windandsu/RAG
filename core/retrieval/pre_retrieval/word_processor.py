from . import BaseDocumentProcessor
import docx
from typing import List, Dict, Any

class WordProcessor(BaseDocumentProcessor):
    def initialize(self):
        pass

    def process(self, file_path: str) -> List[Dict[str, Any]]:
        documents = []
        try:
            doc = docx.Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            content = '\n'.join(full_text)
            if content:
                document = {
                    'content': content,
                    'title': file_path.split('/')[-1],
                    'source': file_path
                }
                documents.append(document)
        except Exception as e:
            print(f"Error processing Word file {file_path}: {e}")
        return documents
from . import BaseDocumentProcessor
from typing import List, Dict, Any

class TXTProcessor(BaseDocumentProcessor):
    def initialize(self):
        pass

    def process(self, file_path: str) -> List[Dict[str, Any]]:
        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                if content:
                    document = {
                        'content': content,
                        'title': file_path.split('/')[-1],
                        'source': file_path
                    }
                    documents.append(document)
        except Exception as e:
            print(f"Error processing TXT file {file_path}: {e}")
        return documents
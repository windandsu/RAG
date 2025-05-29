from . import BaseDocumentProcessor
import PyPDF2
from typing import List, Dict, Any

class PDFProcessor(BaseDocumentProcessor):
    def initialize(self):
        pass

    def process(self, file_path: str) -> List[Dict[str, Any]]:
        documents = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    content = page.extract_text()
                    if content:
                        document = {
                            'content': content,
                            'title': file_path.split('/')[-1],
                            'source': file_path
                        }
                        documents.append(document)
        except Exception as e:
            print(f"Error processing PDF file {file_path}: {e}")
        return documents
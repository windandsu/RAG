from core.retrieval.pre_retrieval.pdf_processor import PDFProcessor
from core.retrieval.pre_retrieval.word_processor import WordProcessor
from core.retrieval.pre_retrieval.txt_processor import TXTProcessor

# 处理PDF文件
pdf_processor = PDFProcessor({})
pdf_documents = pdf_processor.process('F:/RAG文献_gjy/NLP_RAG_Survey.pdf')


# 处理Word文件
word_processor = WordProcessor({})
word_documents = word_processor.process('C:/Users/admin/Desktop/test.docx')

# 处理TXT文件
txt_processor = TXTProcessor({})
txt_documents = txt_processor.process('C:/Users/admin/Desktop/临时文件.txt')


print(pdf_documents)
print(word_documents)
print(txt_documents)
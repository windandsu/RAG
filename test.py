import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from core.retrieval.granularity.chunk.fixed_cutting import FixedChunking
from core.retrieval.granularity.chunk.semantic_cutting import SemanticChunking
from core.retrieval.retrievers.dense_retriever import DenseRetriever

# 示例文档
documents = [
    {'id': 1, 
     'content': '深度学习作为人工智能领域的核心技术，近年来取得了巨大的进展，在图像识别、语音识别、自然语言处理等众多领域展现出了强大的能力。其发展历程充满了突破与创新，经历了多个重要阶段。​深度学习的起源可以追溯到 20 世纪 40 年代。1943 年，心理学家 Warren McCulloch 和数学家 Walter Pitts 提出了 M - P 模型，这是最早的神经网络模型，它基于生物神经元的结构和功能进行建模，通过逻辑运算模拟神经元的激活过程，为后续的神经网络研究奠定了基础。1949 年，心理学家 Donald Hebb 提出 Hebb 学习规则，描述了神经元之间连接强度（即权重）的变化规律，认为神经元之间的连接强度会随着它们之间的活动同步性而增强，为神经网络学习算法提供了重要启示。​1950 年代到 1960 年代，Frank Rosenblatt 提出感知器模型，这是一种简单的神经网络结构，主要用于解决二分类问题。但由于其只能处理线性可分问题，对复杂问题处理能力有限，使得神经网络研究陷入停滞。直到 1986 年，David Rumelhart、Geoffrey Hinton 和 Ron Williams 等科学家提出误差反向传播（Backpropagation）算法，标志着神经网络研究的复兴。在反向传播算法的推动下，多层感知器（MLP）成为多层神经网络的代表，随着计算能力的提升和大数据的普及，基于多层神经网络的深度学习逐渐成为研究热点。​进入深度学习时代，卷积神经网络（CNN）和循环神经网络（RNN）等模型得到广泛应用。CNN 在图像识别领域表现卓越，例如在 ImageNet 图像分类挑战中，深度 CNN 模型的错误率一度降至极低水平，优于人类专家在同一任务上的表现，还广泛应用于安防监控中的人脸识别、自动驾驶中的视觉感知等。RNN 则在自然语言处理和语音识别等领域发挥重要作用，像长短期记忆网络（LSTM）作为 RNN 的变体，有效解决了梯度消失问题，在语音识别中显著提高准确率。​近年来，随着深度学习模型参数和预训练数据规模的不断增加，大模型时代来临。Transformer 最初为自然语言处理任务设计，其核心的自注意力机制能捕捉输入序列中的依赖关系，基于 Transformer 架构的大型预训练语言模型如 BERT、GPT 等在阅读理解、问答、摘要生成等任务上屡创佳绩。基于 Diffusion Model 的 Sora 大模型更是惊艳世人，带领我们进入多模态的人工智能时代。​深度学习算法具有诸多优点。它能通过大量数据学习特征，实现高精度的预测和分类；可通过不断学习和调整参数，自适应不同场景，处理复杂数据结构和非线性问题；还能通过增加层数和节点数解决更复杂问题，并借助分布式计算和 GPU 加速等技术实现高效计算。但深度学习也存在不足，对数据依赖性强，数据质量和规模影响较大；计算资源需求高；模型内部运作机制缺乏解释性，难以解释预测结果及排查错误。​总体来看，深度学习从早期简单模型发展到如今强大的大模型，在众多领域取得了显著成就。未来，深度学习将在模型优化、提高可解释性、降低计算资源需求等方面持续发展，进一步推动人工智能技术的进步，为更多行业带来变革。', 
     'title': '文档1', 
     'source': '示例来源'},
     
]

# 初始化分块器
fixed_config = {'chunk_size': 500, 'overlap': 100}
fixed_chunking = FixedChunking(fixed_config)

semantic_config = {'model_name': 'all-MiniLM-L6-v2', 'similarity_threshold': 0.7, 'min_chunk_size': 200}
semantic_chunking = SemanticChunking(semantic_config)

# 分块操作
chunks = []
for doc in documents:
    fixed_chunks = fixed_chunking.process(doc)
    semantic_chunks = semantic_chunking.process(doc)
    chunks.extend(fixed_chunks)
    chunks.extend(semantic_chunks)

# 初始化嵌入模型
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 计算分块的嵌入
chunk_embeddings = embedding_model.encode([chunk['content'] for chunk in chunks])

# 初始化 Faiss 向量数据库
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# 添加嵌入到 Faiss 数据库
index.add(chunk_embeddings)

# 初始化密集检索器
dense_config = {
    'index_path': 'dense_index',
    'embedding_model': 'all-MiniLM-L6-v2'
}
dense_retriever = DenseRetriever(dense_config)
dense_retriever.document_embeddings = chunk_embeddings
dense_retriever.documents = chunks

# 用户输入查询
query = input("请输入查询内容: ")

# 执行检索
results = dense_retriever.retrieve(query)

# 输出检索结果
print("检索结果:")
for result in results:
    print(f"标题: {result['title']}")
    print(f"内容: {result['content']}")
    print(f"得分: {result['score']}")
    print("-" * 50)
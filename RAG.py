# import fitz  # PyMuPDF
# from tqdm import tqdm

# def load_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     pages = []
#     for i, page in tqdm(enumerate(doc), total=doc.page_count):
#         text = page.get_text()
#         pages.append({"page": i, "text": text})
#     return pages

# pages = load_pdf("human_nutrition_text.pdf")

# from sentence_transformers import SentenceTransformer

# def split_text(text, chunk_size=300):
#     # 简单按字符数切分，可用更智能的分句/分段
#     return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# chunks = []
# for page in pages:
#     for chunk in split_text(page["text"]):
#         if len(chunk.strip()) > 50:  # 过滤过短片段
#             chunks.append({"page": page["page"], "text": chunk})

# import numpy as np

# model = SentenceTransformer("all-mpnet-base-v2")
# embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=True)
# embeddings = np.array(embeddings)

# def search(query, embeddings, chunks, top_k=5):
#     query_vec = model.encode([query])[0]
#     scores = np.dot(embeddings, query_vec)
#     top_indices = np.argsort(scores)[-top_k:][::-1]
#     return [chunks[i] for i in top_indices]

# query = "蛋白质的主要功能是什么？"
# results = search(query, embeddings, chunks)
# for r in results:
#     print(f"Page {r['page']}: {r['text'][:100]}...")

# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_id = "/root/autodl-tmp/DeepSeek-R1"  # 需提前下载/授权
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# llm = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")

# def generate_answer(query, context, max_new_tokens=256):
#     prompt = f"已知信息：\n{context}\n\n请根据上述内容回答：{query}"
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
#     output = llm.generate(input_ids, max_new_tokens=max_new_tokens)
#     return tokenizer.decode(output[0], skip_special_tokens=True)

# context = "\n".join([r["text"] for r in results])
# answer = generate_answer(query, context)
# print(answer)

# def rag_qa(query):
#     results = search(query, embeddings, chunks)
#     context = "\n".join([r["text"] for r in results])
#     return generate_answer(query, context)

# print(rag_qa("水溶性维生素有哪些？"))

import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# 1. 加载Wikipedia数据集
print("加载Wikipedia数据集...")

# 方案A：加载完整的Wikipedia（很大，约20GB）
# dataset = load_dataset("wikipedia", "20220301.en", split="train")

# 方案B：加载Wikipedia的向量化版本（文本+向量）
# 或使用预处理好的数据集
# dataset = load_dataset("embedding-data/wikipedia-22-12-en-embeddings", split="train")

# 方案C：加载小样本测试
# dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

dataset = load_dataset("sentence-transformers/wikipedia-en-sentences", split="train[:10000]")

# 2. 创建文本块
print("处理文本数据...")
chunks = []

if "embeddings" in dataset.column_names:
    # 如果数据集已有向量
    print("使用预计算的向量...")
    chunks = []
    embeddings_list = []
    
    for i, item in tqdm(enumerate(dataset), total=min(10000, len(dataset)), desc="加载数据"):
        if "text" in item and len(item["text"].strip()) > 50:
            chunks.append({"id": i, "text": item["text"][:500]})  # 截断前500字符
        if "embeddings" in item:
            embeddings_list.append(item["embeddings"])
        
        if len(chunks) >= 10000:  # 限制数量，避免内存不足
            break
    
    if embeddings_list:
        embeddings = np.array(embeddings_list[:len(chunks)])
    else:
        # 需要自己计算向量
        model = SentenceTransformer("all-mpnet-base-v2")
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings)
        
else:
    # 普通文本数据集，需要自己分块和向量化
    print("文本数据集，需要分块和向量化...")
    
    def split_text(text, chunk_size=300, overlap=50):
        """更智能的分块"""
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk and len(current_chunk.strip()) > 50:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk and len(current_chunk.strip()) > 50:
            chunks.append(current_chunk.strip())
        return chunks
    
    # 处理数据集
    for i, item in tqdm(enumerate(dataset), total=min(20000, len(dataset)), desc="处理文本"):
        if "text" in item:
            text = item["text"]
            for chunk in split_text(text):
                if len(chunk.strip()) > 50:
                    chunks.append({"id": i, "text": chunk})
        
        if len(chunks) >= 20000:  # 限制数量
            break
    
    # 计算向量
    print("计算文本向量...")
    model = SentenceTransformer("all-mpnet-base-v2")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings)

print(f"数据集大小: {len(chunks)} 个文本块")

# 3. 搜索函数
def search(query, top_k=5):
    query_vec = model.encode([query])[0]
    scores = np.dot(embeddings, query_vec)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# 4. 加载大语言模型
print("加载大语言模型...")
model_id = "/root/autodl-tmp/DeepSeek-R1"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    print("模型加载成功！")
except Exception as e:
    print(f"加载模型失败: {e}")
    # 使用小模型测试
    model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# 5. 生成答案
def generate_answer(query, context, max_new_tokens=256):
    prompt = f"已知信息：\n{context}\n\n请根据上述内容回答：{query}"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = llm.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]

# 6. RAG问答
def rag_qa(query):
    results = search(query)
    print(f"找到相关段落: {len(results)} 个")
    for i, r in enumerate(results[:3]):
        print(f"[{i+1}] {r['text'][:150]}...")
    
    context = "\n".join([r["text"] for r in results])
    return generate_answer(query, context)

# 7. 测试
if __name__ == "__main__":
    test_queries = [
        "什么是人工智能？",
        "爱因斯坦的主要贡献是什么？",
        "巴黎位于哪个国家？",
        "机器学习的基本原理是什么？"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"问题: {query}")
        print(f"{'='*60}")
        answer = rag_qa(query)
        print(f"\n回答: {answer}")
        print(f"{'='*60}\n")
# knowledge_engine.py

import json
from sentence_transformers import SentenceTransformer, util

# ─── โหลด JSON และเตรียม topics_data ────────────────────
with open("buddhist_knowledge.json", "r", encoding="utf-8") as f:
    data = json.load(f)
topics_data = data["topics"]

# ─── สร้าง embeddings สำหรับ semantic lookup ────────────────
model = SentenceTransformer("all-MiniLM-L6-v2")
knowledge_texts = [topic["title"] + " " + topic["content"] for topic in topics_data]
embeddings = model.encode(knowledge_texts, convert_to_tensor=True)

def get_answer_from_knowledge(user_question: str) -> dict:
    """
    หา topic ที่ใกล้เคียงที่สุดกับ user_question
    แล้ว return dict: {"title": ..., "content": ...}
    """
    user_emb = model.encode(user_question, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(user_emb, embeddings)[0]
    best_idx = int(scores.argmax().item())
    best_topic = topics_data[best_idx]
    return {
        "title":   best_topic["title"],
        "content": best_topic["content"]
    }

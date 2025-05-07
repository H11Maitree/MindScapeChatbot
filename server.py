import os
import random
import json
from flask import Flask, request, jsonify, render_template
from langdetect import detect, LangDetectException
import openai

from knowledge_engine import get_answer_from_knowledge

# ─── ตั้งค่า API Key จาก ENV ─────────────────────────────────
# ก่อนรัน ให้ตั้ง ENV var ชื่อ OPENAI_API_KEY (ไม่ต้องใส่เครื่องหมายคำพูด)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY")

# ─── Few-shot Prompt สำหรับ Classification ────────────────────
CLASSIFY_PROMPT = """
You are a classifier.  Classify the user request into exactly one of these 4 categories:
1) GREETING         (e.g. hi, สวัสดี, hello)
2) DHAMMA_TOPIC     (questions about Buddhist teachings, topics, keywords)
3) EMOTIONAL_SUPPORT (life or emotional problems seeking encouragement or listening)
4) OTHER            (all other topics: cooking, tech support, travel tips, recipes, etc.)

Only return the single word: GREETING, DHAMMA_TOPIC, EMOTIONAL_SUPPORT, or OTHER
(with no extra text).
"""

# ─── Pool ของประโยคทักทาย (สุ่มภาษาไทย/อังกฤษ) ────────────────
GREETINGS = {
    "th": [
        "เฮ้ย สวัสดี! วันนี้เป็นไงบ้าง 😊 มีอะไรในใจอยากเล่าให้ฟังกันมั้ย?",
        "หวัดดีจ้า! เหนื่อย ๆ รึเปล่า มาคุยเรื่องหัวใจหรือธรรมะชิล ๆ กัน~",
        "ไงเพื่อน! วันนี้อยากปรึกษาอะไรหรือให้เราเติมพลังบวกด้วยธรรมะบ้างไหม?"
    ],
    "en": [
        "Hey there! How’s it going today? 😊 Anything on your mind you’d like to share?",
        "Hi friend! Feeling up or down? Let’s chat—happy to lend an ear and some positive Dhamma vibes.",
        "Yo! What’s on your heart today? I’m here to listen and sprinkle some uplifting Buddhist wisdom."
    ]
}

# ─── ข้อความปฏิเสธกรณี OTHER ────────────────────────────────
DENIAL = {
    "th": "ขอโทษนะ ฉันช่วยได้เฉพาะเรื่องธรรมะหรือปรึกษาชีวิต-อารมณ์เท่านั้นค่ะ 🙏",
    "en": "Sorry, I only discuss Dhamma topics or offer emotional support. 😊"
}

# ─── System prompts สำหรับ ChatCompletion ─────────────────────
SYSTEM_SUPPORT = {
    "th": "คุณคือเพื่อนพลังบวกสายธรรมะ ชิล ๆ เข้าใจชีวิตและอารมณ์ พร้อมอ้างอิงหลักธรรมมาช่วยให้กำลังใจ",
    "en": "You are a laid-back Dhamma buddy who listens warmly, offers emotional support, and cites Buddhist teachings."
}
SYSTEM_DHAMMA = {
    "th": "คุณคือผู้เชี่ยวชาญด้านธรรมะ ให้คำอธิบายคอนเทนต์ธรรมะที่มาจาก knowledge base อย่างชัดเจน เข้าใจง่าย",
    "en": "You are an expert in Buddhist Dhamma, explaining topics clearly based on the provided knowledge snippets."
}

# ─── Flask App Setup ────────────────────────────────────────
app = Flask(__name__)

def detect_lang(text: str) -> str:
    """Detect language code 'th' or 'en'"""
    try:
        return "th" if detect(text) == "th" else "en"
    except LangDetectException:
        return "th"

def classify_request(user_msg: str) -> str:
    """ส่งข้อความให้ OpenAI แยกประเภทเป็น 4 กรณี"""
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": CLASSIFY_PROMPT},
            {"role": "user",   "content": user_msg}
        ],
        temperature=0
    )
    return resp.choices[0].message.content.strip().upper()

@app.route("/", methods=["GET"])
def index():
    return render_template("Chatbot.html")

@app.route("/chat", methods=["POST"])
def chat():
    payload      = request.get_json()
    user_message = payload.get("message", "").strip()
    history      = payload.get("history", [])   # [{"role":"user"/"assistant","content":...}, ...]
    lang         = detect_lang(user_message)

    # 1) Classification
    category = classify_request(user_message)

    # 2) GREETING → ตอบสุ่มจาก GREETINGS
    if category == "GREETING":
        reply = random.choice(GREETINGS.get(lang, GREETINGS["en"]))
        return jsonify({"reply": reply})

    # 3) OTHER → ปฏิเสธ
    if category == "OTHER":
        return jsonify({"reply": DENIAL.get(lang)})

    # 4) DHAMMA_TOPIC → ดึงจาก knowledge base แล้ว refine ผ่าน OpenAI
    if category == "DHAMMA_TOPIC":
        raw = get_answer_from_knowledge(user_message)

        # ─── แปลง raw ให้เป็นข้อความเสมอ ────────────────────
        if isinstance(raw, dict):
            snippet = raw.get("content", "")
        else:
            snippet = str(raw)

        # ถ้า snippet ยังว่าง → แจ้งไม่มีข้อมูล
        if not snippet:
            return jsonify({"reply": {
                "th": "ขอโทษค่ะ ยังไม่มีเนื้อหาธรรมะในเรื่องนี้ 🤍",
                "en": "Sorry, I don't have Dhamma material on that topic yet 🤍"
            }[lang]})

        # ส่ง snippet + คำสั่ง refine ไปให้ OpenAI ปรับให้ลื่นไหล
        refine = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_DHAMMA[lang]},
                {"role": "user",   "content": snippet},
                {"role": "user",   "content": f"ช่วยปรับประโยคให้กระชับ ลื่นไหล และตอบให้ตรงคำถามนี้: {user_message}"}
            ],
            temperature=0.7
        )
        return jsonify({"reply": refine.choices[0].message.content.strip()})

    # 5) EMOTIONAL_SUPPORT → ส่งพร้อม history
    if category == "EMOTIONAL_SUPPORT":
        messages = [{"role":"system","content":SYSTEM_SUPPORT[lang]}] + history
        messages.append({"role":"user","content":user_message})
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.8
        )
        return jsonify({"reply": resp.choices[0].message.content.strip()})

    # Safety net
    return jsonify({"reply": DENIAL.get(lang)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)

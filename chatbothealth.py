# chatbothealth.py
import os
import sys
from datetime import datetime
from flask import Flask, request, render_template, jsonify
from twilio.twiml.messaging_response import MessagingResponse

import requests
import json
import random
import nltk
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- CONFIG ----------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_FILE = os.path.join(APP_DIR, "health_faq.txt")
USER_AGENT = {"User-Agent": "Mozilla/5.0 (compatible; HealthBot/1.0)"}
TIMEOUT = 10  # seconds for HTTP requests
# ---------------------------

app = Flask(__name__)

# -----------------------------
# Load FAQ (Q/A pairs) from file
def load_faq(filepath=FAQ_FILE):
    if not os.path.exists(filepath):
        print(f"[ERROR] health_faq.txt not found at: {filepath}", file=sys.stderr)
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    faqs = []
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            q = lines[i].replace("Q:", "").strip().lower()
            a = lines[i + 1].replace("A:", "").strip()
            faqs.append((q, a))
    return faqs

faq_pairs = load_faq()
questions = [q for q, _ in faq_pairs]
answers = [a for _, a in faq_pairs]

# -----------------------------
# NLTK preprocessing
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
lemmer = nltk.stem.WordNetLemmatizer()
remove_punc_dict = dict((ord(p), None) for p in string.punctuation)

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

# -----------------------------
# Greeting helpers
GREET_IN = ('hello', 'hi', 'hey', 'namaste', 'whassup', 'how are you?')
GREET_OUT = ('Hello!', 'Hi there!', 'Hey!', 'Namaste!')

def maybe_greet(txt: str):
    for w in txt.split():
        if w.lower() in GREET_IN:
            return random.choice(GREET_OUT)
    return None

# -----------------------------
# Real-time data integrations

def get_covid_update_india():
    """
    Uses disease.sh public API for India summary (simple and reliable).
    """
    url = "https://disease.sh/v3/covid-19/countries/india"
    try:
        r = requests.get(url, headers=USER_AGENT, timeout=TIMEOUT)
        if r.status_code != 200:
            return "Live COVID update is temporarily unavailable."
        j = r.json()
        today_cases = j.get("todayCases", "N/A")
        today_deaths = j.get("todayDeaths", "N/A")
        total_cases = j.get("cases", "N/A")
        updated = j.get("updated")
        when = datetime.fromtimestamp(updated/1000).strftime("%d-%b-%Y %H:%M") if updated else "N/A"
        return (f"🇮🇳 India COVID Update:\n"
                f"• New cases today: {today_cases}\n"
                f"• New deaths today: {today_deaths}\n"
                f"• Total cases: {total_cases}\n"
                f"(Last updated: {when})")
    except Exception:
        return "Live COVID update is temporarily unavailable."

def get_who_outbreak_headlines(limit=5):
    """
    Reads WHO Disease Outbreak News RSS titles (top N).
    """
    import xml.etree.ElementTree as ET
    url = "https://www.who.int/feeds/entity/csr/don/en/rss.xml"
    try:
        r = requests.get(url, headers=USER_AGENT, timeout=TIMEOUT)
        if r.status_code != 200:
            return "WHO outbreak feed is unavailable right now."
        root = ET.fromstring(r.content)
        # RSS -> channel -> item -> title
        channel = root.find("channel")
        items = channel.findall("item") if channel is not None else []
        if not items:
            return "No current outbreak items from WHO."
        titles = []
        for it in items[:limit]:
            t = it.find("title").text if it.find("title") is not None else None
            if t:
                titles.append(f"• {t}")
        return "🆘 WHO Disease Outbreak News:\n" + "\n".join(titles) if titles else "No current outbreak items from WHO."
    except Exception:
        return "WHO outbreak feed is unavailable right now."

def get_cowin_slots_by_pin(pincode: str, date_ddmmyyyy: str):
    """
    Optional: CoWIN public sessions API by PIN (may throttle/limit).
    Example date format: '31-08-2025'
    """
    url = f"https://cdn-api.co-vin.in/api/v2/appointment/sessions/public/findByPin?pincode={pincode}&date={date_ddmmyyyy}"
    try:
        r = requests.get(url, headers=USER_AGENT, timeout=TIMEOUT)
        if r.status_code != 200:
            return "Vaccination slot service is temporarily unavailable."
        data = r.json().get("sessions", [])
        if not data:
            return "No vaccination slots found for the given PIN and date."
        lines = ["💉 Available Vaccination Slots:"]
        for s in data[:10]:
            lines.append(
                f"• {s.get('name','Center')} | {s.get('vaccine','Vaccine')} | "
                f"Min Age: {s.get('min_age_limit','-')} | Dose1: {s.get('available_capacity_dose1','-')} | "
                f"Dose2: {s.get('available_capacity_dose2','-')}"
            )
        return "\n".join(lines)
    except Exception:
        return "Vaccination slot service is temporarily unavailable."

# -----------------------------
# Core Response Logic

def faq_response(user_text: str, threshold: float = 0.35):
    """
    TF-IDF similarity over FAQ questions.
    """
    if not questions:
        return None
    user_text = user_text.lower().strip()
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(questions + [user_text])
    sims = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = sims.argsort()[0][-1]
    score = sims[0][idx]
    if score < threshold:
        return None
    return answers[idx]

def handle_user_message(user_msg: str) -> str:
    if not user_msg:
        return "Please type your question."

    # Greetings
    g = maybe_greet(user_msg)
    if g:
        return f"{g} Ask me about symptoms, prevention, vaccines, or outbreaks."

    text = user_msg.lower().strip()

    # Real-time intents (simple keyword triggers for demo)
    if "covid" in text or "corona" in text:
        return get_covid_update_india()

    if "who" in text and ("outbreak" in text or "news" in text or "alert" in text):
        return get_who_outbreak_headlines()

    if ("vaccination" in text or "vaccine" in text) and ("slot" in text or "available" in text or "pin" in text):
        # very simple parser: look for 6-digit PIN and dd-mm-yyyy date
        import re
        pin_match = re.search(r"\b(\d{6})\b", text)
        date_match = re.search(r"\b(\d{2}-\d{2}-\d{4})\b", text)
        if pin_match and date_match:
            return get_cowin_slots_by_pin(pin_match.group(1), date_match.group(1))
        else:
            return "To check vaccination slots, send: 'Check slots PINCODE DATE' (e.g., 'Check slots 700001 31-08-2025')."

    # FAQ (static knowledge base)
    ans = faq_response(text)
    if ans:
        return ans

    # Fallback
    return "Sorry, I don’t have that information yet. Please consult a healthcare professional."

# -----------------------------
# Routes

@app.route("/")
def home():
    return render_template("index.html")

# JSON endpoint for the frontend
@app.route("/get", methods=["POST"])
def get_reply():
    try:
        payload = request.get_json(force=True)
        msg = payload.get("message", "")
    except Exception:
        msg = request.form.get("message", "")  # fallback if form-encoded

    reply = handle_user_message(msg)
    return jsonify({"reply": reply})

# WhatsApp webhook (Twilio)
@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    user_message = request.form.get("Body", "")
    reply = handle_user_message(user_message)
    tw = MessagingResponse()
    tw.message(reply)
    return str(tw)

if __name__ == "__main__":
    print(f"[INFO] Looking for FAQ at: {FAQ_FILE}")
    app.run(debug=True)

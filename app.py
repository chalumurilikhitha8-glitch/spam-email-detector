import streamlit as st
import pickle
import re
import json
from datetime import datetime


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s@._:/-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_links(text):
    return re.findall(r"(https?://\S+|www\.\S+)", text.lower())


def calculate_risk_score(found_words, found_links, found_domains, urgent_patterns):
    score = 0
    score += len(found_words) * 8
    score += len(found_links) * 15
    score += len(found_domains) * 25
    score += len(urgent_patterns) * 10
    return min(score, 100)


def risk_level(score):
    if score >= 70:
        return "High Risk"
    if score >= 40:
        return "Medium Risk"
    return "Low Risk"


def generate_explanation(prediction, reasons, risk_score):
    if prediction == "spam":
        return f"This email is HIGH RISK.\n\nReasons:\n- " + "\n- ".join(reasons)
    else:
        return f"This email looks SAFE.\nRisk Score: {risk_score}%"


st.set_page_config(page_title="Spam Email Detector AI", page_icon="📧")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("📧 Spam Email Detector AI")

sender = st.text_input("Sender Email")
subject = st.text_input("Email Subject")
body = st.text_area("Email Body", height=200)

if st.button("Detect Spam"):

    full_text = clean_text(sender + " " + subject + " " + body)
    sender_clean = clean_text(sender)

    vec = vectorizer.transform([full_text])
    model_prediction = model.predict(vec)[0]

    model_confidence = None
    if hasattr(model, "predict_proba"):
        model_confidence = float(model.predict_proba(vec).max())

    suspicious_words = [
        "urgent", "free", "click", "password", "verify",
        "bank", "claim", "login", "account", "reset"
    ]

    suspicious_domains = ["@paypa1.com", "@secure-bank.net"]

    found_words = [w for w in suspicious_words if w in full_text]
    found_domains = [d for d in suspicious_domains if d in sender_clean]

    extracted_links = extract_links(body)
    found_links = extracted_links if len(extracted_links) > 2 else []

    if found_words or found_domains or found_links:
        final_prediction = "spam"
    else:
        final_prediction = model_prediction

    risk_score = calculate_risk_score(found_words, found_links, found_domains, [])

    reasons = []
    if found_words:
        reasons.append("Suspicious words: " + ", ".join(found_words))
    if found_domains:
        reasons.append("Fake domain detected")
    if found_links:
        reasons.append("Too many links")

    if not reasons:
        reasons.append("Based on AI model")

    st.subheader("Result")

    if final_prediction == "spam":
        st.error("🚨 SPAM DETECTED")
    else:
        st.success("✅ SAFE EMAIL")

    st.write(f"Prediction: {final_prediction}")
    if model_confidence:
        st.write(f"Confidence: {round(model_confidence*100,2)}%")

    st.write(f"Risk Score: {risk_score}%")

    st.write("Reasons:")
    for r in reasons:
        st.write("-", r)

    st.markdown("### 🤖 AI Explanation")
    st.info(generate_explanation(final_prediction, reasons, risk_score))

    report = {
        "sender": sender,
        "subject": subject,
        "prediction": final_prediction,
        "risk": risk_score,
        "reasons": reasons,
        "time": str(datetime.now())
    }

    st.download_button("Download Report", json.dumps(report, indent=2), "report.json")

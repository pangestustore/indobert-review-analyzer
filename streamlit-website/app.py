import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load model sentimen
sentiment_model_path = "pangestuu/indobert_sentiment_aspek"
tokenizer_sentiment = AutoTokenizer.from_pretrained(sentiment_model_path)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)
sentiment_labels = ['Negative', 'Positive']

# Load model aspek
aspect_model_path = "pangestuu/indobert_sentiment_aspek"
tokenizer_aspect = AutoTokenizer.from_pretrained(aspect_model_path)
model_aspect = AutoModelForSequenceClassification.from_pretrained(aspect_model_path)

# Label aspek sesuai yang kamu berikan
aspect_labels = [
    "Akses Layanan Kesehatan Online",
    "Kemudahan Akses dan Kinerja Aplikasi",
    "Kendala Login dan Pembaruan",
    "Kendala Verifikasi dan OTP",
    "Kesulitan Penggunaan Aplikasi",
    "Manajemen Data dan Faskes"
]

def predict(text, tokenizer, model, labels):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()
    return labels[predicted_class], confidence

# Streamlit App
st.set_page_config(page_title="Analisis Ulasan Mobile JKN", layout="centered")
st.title("🩺 Analisis Ulasan Mobile JKN")
st.markdown("Masukkan ulasan dari pengguna aplikasi untuk memprediksi **aspek** dan **sentimen**.")

text = st.text_area("📝 Masukkan Ulasan Pengguna Aplikasi")

if st.button("🔍 Analisis"):
    if not text.strip():
        st.warning("Tolong masukkan teks ulasan terlebih dahulu.")
    else:
        aspect, conf_aspect = predict(text, tokenizer_aspect, model_aspect, aspect_labels)
        sentiment, conf_sent = predict(text, tokenizer_sentiment, model_sentiment, sentiment_labels)

        st.subheader("📊 Hasil Analisis:")
        st.markdown(f"- **Aspek**: `{aspect}` (Confidence: {conf_aspect:.2f})")
        st.markdown(f"- **Sentimen**: `{sentiment}` (Confidence: {conf_sent:.2f})")

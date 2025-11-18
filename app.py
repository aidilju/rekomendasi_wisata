import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import pickle
import os
from flask import Flask, render_template, request, jsonify

# --- KONSTANTA ---
MODEL_FILE = 'rekomendasi_model.pkl'
TOP_N = 5
ALPHA = 0.5

# --- FUNGSI PREPROCESSING ---
nltk.download('stopwords', quiet=True)
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('indonesian'))
    lokasi_khusus = {'barat', 'timur', 'utara', 'selatan', 'tengah'}
    stop_words = stop_words - lokasi_khusus
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

def safe_normalize(scores):
    if len(scores) == 0: return scores
    min_val = np.min(scores)
    max_val = np.max(scores)
    delta = max_val - min_val
    if delta < 1e-9:
        return np.zeros_like(scores)
    return (scores - min_val) / delta

# --- LOAD MODEL ---
data = None
vectorizer = None
tfidf_matrix = None
bm25 = None

if os.path.exists(MODEL_FILE):
    print(f"Memuat model '{MODEL_FILE}'...")
    try:
        with open(MODEL_FILE, 'rb') as f:
            loaded_model = pickle.load(f)
        data = loaded_model['data_full'] 
        vectorizer = loaded_model['vectorizer']
        tfidf_matrix = loaded_model['tfidf_matrix']
        bm25 = loaded_model['bm25']
        print(f"Model BERHASIL dimuat.")
    except Exception as e:
        print(f"Gagal memuat model: {e}")
else:
    print(f"PERINGATAN: File '{MODEL_FILE}' TIDAK DITEMUKAN! Jalankan buat_model.py dulu.")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if data is None:
        return jsonify({"error": "Model belum siap."}), 500

    kueri_pengguna = request.json.get('query')
    if not kueri_pengguna:
        return jsonify({"error": "Kueri kosong"}), 400

    query_processed = preprocess(kueri_pengguna)
    tokenized_query = query_processed.split()

    if not tokenized_query:
        return jsonify([])

    # Hitung Skor
    query_vector = vectorizer.transform([query_processed])
    vsm_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Normalisasi & Hybrid
    vsm_norm = safe_normalize(vsm_scores)
    bm25_norm = safe_normalize(bm25_scores)
    hybrid_scores = ALPHA * vsm_norm + (1 - ALPHA) * bm25_norm
    
    top_indices = hybrid_scores.argsort()[::-1][:TOP_N]

    results = []
    for idx in top_indices:
        if hybrid_scores[idx] <= 0: continue
        row = data.iloc[idx]
        results.append({
            "id": int(row['FID']),
            "nama_objek": row['nama_objek'],
            "provinsi": row['provinsi'],
            "alamat": str(row['alamat']),
            "deskripsi": str(row['deskripsi']),
            "skor": float(hybrid_scores[idx])
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
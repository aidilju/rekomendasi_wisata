import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
# Menggunakan joblib untuk memuat file yang dibuat joblib.dump
import joblib 
import os
from flask import Flask, render_template, request, jsonify

# --- KONSTANTA ---
MODEL_FILE = 'rekomendasi_model.pkl'
TOP_N = 5
ALPHA = 0.5

# --- FUNGSI PREPROCESSING ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(text):
    """Membersihkan dan men-stemming teks kueri."""
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
    """Normalisasi skor ke rentang 0-1."""
    if len(scores) == 0: return scores
    min_val = np.min(scores)
    max_val = np.max(scores)
    delta = max_val - min_val
    if delta < 1e-9:
        return np.zeros_like(scores)
    return (scores - min_val) / delta

# --- LOAD MODEL (Perbaikan utama di bagian ini) ---
data = None
vectorizer = None
tfidf_matrix = None
bm25 = None

if os.path.exists(MODEL_FILE):
    print(f"Memuat model '{MODEL_FILE}'...")
    try:
        # PERBAIKAN 1: Menggunakan joblib.load() untuk file yang disimpan oleh joblib
        loaded_model = joblib.load(MODEL_FILE) 
        
        # PERBAIKAN 2: Menggunakan kunci 'data_df' (sesuai build_model.py)
        data = loaded_model['data_df'] 
        
        vectorizer = loaded_model['vectorizer']
        tfidf_matrix = loaded_model['tfidf_matrix']
        bm25 = loaded_model['bm25']
        print(f"Model BERHASIL dimuat.")
    except Exception as e:
        print(f"Gagal memuat model. Error: {e}")
else:
    print(f"PERINGATAN: File '{MODEL_FILE}' TIDAK DITEMUKAN! Jalankan buat_model.py dulu.")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if data is None or vectorizer is None or bm25 is None:
        return jsonify({"error": "Model belum siap. Model perlu dimuat dengan sukses sebelum rekomendasi dapat diberikan."}), 500

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
    
    # BM25 membutuhkan token kueri
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
        
        # PERHATIAN: Asumsi nama kolom yang benar adalah 'nama_destinasi'
        nama_objek = row.get('nama_destinasi', 'Nama Destinasi Tidak Tersedia')

        results.append({
            "id": int(row['FID']),
            "nama_objek": nama_objek, 
            "provinsi": row['provinsi'],
            "alamat": str(row['alamat']),
            "deskripsi": str(row['deskripsi']),
            "skor": float(hybrid_scores[idx])
        })

    return jsonify(results)

if __name__ == '__main__':
    # Pastikan nltk.download hanya dijalankan sekali di awal
    app.run(debug=True, host='0.0.0.0', port=5000)

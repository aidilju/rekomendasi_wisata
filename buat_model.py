import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import joblib
import os
import time

# --- KONFIGURASI ---
print("1. Mempersiapkan Library & Sastrawi...")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(text):
    """Membersihkan teks untuk algoritma."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) 
    words = text.split()
    
    stop_words = set(stopwords.words('indonesian'))
    # Kita simpan arah mata angin karena penting untuk lokasi
    lokasi_khusus = {'barat', 'timur', 'utara', 'selatan', 'tengah'}
    stop_words = stop_words - lokasi_khusus
    
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

def combine_text_features(df, columns):
    return df[columns].astype(str).agg(' '.join, axis=1)

def build_and_save_model():
    start_time = time.time()
    path = 'DataPariwisata.xlsx'
    
    if not os.path.exists(path):
        print(f"[ERROR] File '{path}' tidak ditemukan!")
        return

    # 1. Baca Data
    print("2. Membaca Excel...")
    data = pd.read_excel(path)
    # Memastikan hanya kolom yang ada di data Anda
    data = data[['FID', 'provinsi', 'nama_objek', 'alamat', 'deskripsi']]

    # 2. Preprocessing (Berat di sini)
    print("3. Melakukan Preprocessing (Stemming)... Harap bersabar.")
    data['text_full'] = combine_text_features(data, ['provinsi', 'alamat', 'deskripsi'])
    data['deskripsi_clean'] = data['text_full'].apply(preprocess)

    # 3. Latih Model
    print("4. Melatih Model TF-IDF & BM25...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['deskripsi_clean'])

    tokenized_corpus = [doc.split() for doc in data['deskripsi_clean']]
    bm25 = BM25Okapi(tokenized_corpus)

    # 4. Simpan Model
    print("5. Menyimpan ke 'model.joblib'...")
    package = {
        'data_df': data,
        'vectorizer': vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'bm25': bm25
    }
    joblib.dump(package, 'model.joblib')
    
    print(f"\n[SUKSES] Model selesai dibuat dalam {time.time() - start_time:.2f} detik.")
    print("Sekarang jalankan 'python app.py'.")

if __name__ == "__main__":
    build_and_save_model()
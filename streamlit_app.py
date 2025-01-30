import streamlit as st
import pandas as pd
import re
import requests
from collections import Counter
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import numpy as np  # Pour la médiane

# Télécharger les stopwords de nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('french'))

# Charger un modèle NLP pour la similarité sémantique
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Longueur de l'article cible fixée à 1500 mots
ARTICLE_LENGTH = 1500

def fetch_html(url):
    """Récupère le contenu HTML d'une URL"""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return str(soup.body)
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération de {url} : {e}")
        return ""

def extract_text_from_html(html_content):
    """Extrait le texte brut d'un contenu HTML"""
    soup = BeautifulSoup(html_content, "html.parser")
    return ' '.join(soup.stripped_strings)

def get_word_frequencies(text):
    """Calcule la fréquence des mots avec normalisation"""
    words = re.findall(r'\b\w{3,}\b', text.lower())
    words = [word for word in words if word not in stop_words]

    total_words = len(words) if words else 1  # Évite division par 0
    word_freq = Counter(words)

    # Normalisation par 1000 mots
    normalized_freq = {word: (count / total_words) * 1000 for word, count in word_freq.items()}
    return Counter(normalized_freq)

def normalize_frequencies(counter_list):
    """Combine les fréquences de plusieurs sites"""
    combined = Counter()
    for counter in counter_list:
        combined.update(counter)
    return combined

def filter_semantic_words(keyword, word_frequencies):
    """Filtre les mots selon leur similarité sémantique avec le mot-clé cible"""
    keyword_embedding = model.encode(keyword, convert_to_tensor=True)
    filtered_words = {}

    for word, freq in word_frequencies.items():
        word_embedding = model.encode(word, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(keyword_embedding, word_embedding).item()
        if similarity > 0.5:  # Seuil de similarité
            filtered_words[word] = freq

    return Counter(filtered_words)

def evaluate_content(frequencies, reference_frequencies):
    """Calcule un score SEO basé sur la présence des mots-clés"""
    score = 0
    for word, ref_count in reference_frequencies.items():
        if word in frequencies:
            score += min(frequencies[word], ref_count)
    return round((score / sum(reference_frequencies.values())) * 100, 2)

def calculate_recommended_frequencies(word_frequencies_list):
    """Calcule la fréquence médiane des mots et ajuste pour 1500 mots"""
    word_occurrences = {}

    for freq_dict in word_frequencies_list:
        for word, freq in freq_dict.items():
            if word not in word_occurrences:
                word_occurrences[word] = []
            word_occurrences[word].append(freq)

    recommended_frequencies = {
        word: round(np.median(freq_list) * (ARTICLE_LENGTH / 1000))
        for word, freq_list in word_occurrences.items()
    }

    return Counter(recommended_frequencies)

def main():
    st.title("Analyse Sémantique pour le SEO")
    
    keyword = st.text_input("Mot-clé cible :", "")

    st.subheader("Collez les 10 URLs des sites concurrents (une par ligne)")
    urls_input = st.text_area("Entrez les URLs ici")

    if st.button("Analyser"):
        url_list = list(filter(None, map(str.strip, urls_input.split("\n"))))  # Nettoyage des URLs

        if not keyword or len(url_list) < 3:
            st.warning("Veuillez entrer un mot-clé et au moins 3 URLs.")
            return

        # Récupération et extraction de texte
        html_contents = [fetch_html(url) for url in url_list]
        texts = [extract_text_from_html(html) for html in html_contents]
        
        # Fréquences des mots
        frequencies = [get_word_frequencies(text) for text in texts]

        # Génération de la liste sémantique
        raw_ref_frequencies = normalize_frequencies(frequencies)
        ref_frequencies = filter_semantic_words(keyword, raw_ref_frequencies)

        # Évaluation des sites
        scores = [evaluate_content(freq, ref_frequencies) for freq in frequencies]

        # Calcul des fréquences conseillées pour l'article de 1500 mots
        recommended_freq = calculate_recommended_frequencies(frequencies)

        # Affichage des scores des sites
        st.subheader("Scores des sites analysés")
        for i, (url, score) in enumerate(zip(url_list, scores)):
            st.write(f"🔗 **{url}** - Score SEO : {score}/100")

        # Création du dataframe
        df = pd.DataFrame(list(recommended_freq.items()), columns=["Mot", "Fréquence conseillée"])
        df = df.sort_values(by="Fréquence conseillée", ascending=False)

        # Affichage du tableau
        st.subheader("Liste des mots à ajouter à votre article")
        st.dataframe(df)

        # Export CSV
        csv_file = "analyse_semantique.csv"
        df.to_csv(csv_file, index=False)
        with open(csv_file, "rb") as f:
            st.download_button("Télécharger le fichier CSV", f, file_name="analyse_semantique.csv", mime="text/csv")

if __name__ == "__main__":
    main()

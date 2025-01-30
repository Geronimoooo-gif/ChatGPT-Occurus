import streamlit as st
import pandas as pd
import re
import requests
from collections import Counter
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util

# Télécharger les stopwords de nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('french'))

# Charger un modèle NLP pour la similarité sémantique
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def fetch_html(url):
    """ Récupère le HTML d'une URL """
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return str(soup.body)
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération de {url} : {e}")
        return ""

def extract_text_from_html(html_content):
    """ Extrait le texte brut d'un contenu HTML """
    soup = BeautifulSoup(html_content, "html.parser")
    return ' '.join(soup.stripped_strings)

def get_ngrams(text, n=2):
    """ Génère des expressions de n mots (bigrams, trigrams) """
    words = re.findall(r'\b\w{3,}\b', text.lower())  # Exclut les mots courts
    words = [word for word in words if word not in stop_words]  # Supprime les stopwords
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    return Counter(ngrams)

def get_word_frequencies(text):
    """ Calcule la fréquence des mots + bigrams + trigrams """
    words = re.findall(r'\b\w{3,}\b', text.lower())
    words = [word for word in words if word not in stop_words]
    unigram_freq = Counter(words)
    bigram_freq = get_ngrams(text, 2)
    trigram_freq = get_ngrams(text, 3)
    return unigram_freq + bigram_freq + trigram_freq  # Combine tout

def normalize_frequencies(counter_list):
    """ Normalise les fréquences sur plusieurs sources """
    combined = Counter()
    for counter in counter_list:
        combined.update(counter)
    return combined

def filter_semantic_words(keyword, word_frequencies):
    """ Filtre les mots selon leur similarité sémantique avec le mot-clé """
    keyword_embedding = model.encode(keyword, convert_to_tensor=True)
    filtered_words = {}

    for word, freq in word_frequencies.items():
        word_embedding = model.encode(word, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(keyword_embedding, word_embedding).item()
        if similarity > 0.4:  # Seuil élargi pour plus de diversité
            filtered_words[word] = freq
    
    return Counter(filtered_words)

def calculate_presence_rate(ref_frequencies, site_frequencies_list):
    """ Calcule le pourcentage de présence d'un mot parmi les sites analysés """
    presence_count = Counter()
    total_sites = len(site_frequencies_list)

    for word in ref_frequencies.keys():
        count = sum(1 for site_freq in site_frequencies_list if word in site_freq)
        presence_count[word] = round((count / total_sites) * 100, 2)  # En pourcentage

    return presence_count

def main():
    st.title("Analyse Sémantique SEO - 10 URLs")

    keyword = st.text_input("Mot-clé cible :", "")

    st.subheader("Entrez les URLs des 10 premiers sites de la SERP (une par ligne)")
    urls = st.text_area("Copiez-collez jusqu'à 10 URLs :", "").strip().split("\n")
    urls = [url.strip() for url in urls if url.strip()][:10]  # Nettoie et limite à 10

    if st.button("Analyser"):
        if not keyword or len(urls) == 0:
            st.warning("Veuillez entrer un mot-clé et au moins une URL.")
            return

        # Récupération des contenus des sites
        site_texts = [extract_text_from_html(fetch_html(url)) for url in urls]

        # Extraction des mots-clés et expressions
        site_frequencies = [get_word_frequencies(text) for text in site_texts]

        # Normalisation des fréquences globales
        raw_ref_frequencies = normalize_frequencies(site_frequencies)

        # Filtrage sémantique
        ref_frequencies = filter_semantic_words(keyword, raw_ref_frequencies)

        # Calcul du taux de présence des mots
        presence_rates = calculate_presence_rate(ref_frequencies, site_frequencies)

        # Construction du dataframe final
        df = pd.DataFrame(ref_frequencies.most_common(30), columns=["Mot/Expression", "Fréquence"])
        df["Taux de présence (%)"] = df["Mot/Expression"].map(presence_rates)

        # Résultats
        st.subheader("Liste des mots et expressions à ajouter à votre article")
        st.dataframe(df)

        # Export en CSV
        csv_file = "analyse_semantique.csv"
        df.to_csv(csv_file, index=False)
        with open(csv_file, "rb") as f:
            st.download_button("Télécharger le fichier CSV", f, file_name="analyse_semantique.csv", mime="text/csv")

if __name__ == "__main__":
    main()

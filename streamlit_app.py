import streamlit as st
import pandas as pd
import re
import requests
from collections import Counter
from bs4 import BeautifulSoup
import csv

def fetch_html(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return str(soup.body)
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération de {url} : {e}")
        return ""

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return ' '.join(soup.stripped_strings)

def get_word_frequencies(text):
    words = re.findall(r'\b\w{3,}\b', text.lower())  # Exclut les mots de moins de 3 lettres
    return Counter(words)

def normalize_frequencies(counter_list):
    combined = Counter()
    for counter in counter_list:
        combined.update(counter)
    return combined

def evaluate_content(frequencies, reference_frequencies):
    score = 0
    for word, ref_count in reference_frequencies.items():
        if word in frequencies:
            score += min(frequencies[word], ref_count)
    return round((score / sum(reference_frequencies.values())) * 100, 2)

def main():
    st.title("Analyse Sémantique pour le SEO")
    
    keyword = st.text_input("Mot-clé cible :", "")
    
    st.subheader("Entrez les URLs des 3 premiers sites de la SERP")
    url1 = st.text_input("URL du site en position 1")
    url2 = st.text_input("URL du site en position 2")
    url3 = st.text_input("URL du site en position 3")
    
    if st.button("Analyser"):
        if not keyword or not url1 or not url2 or not url3:
            st.warning("Veuillez remplir tous les champs.")
            return
        
        # Récupération du HTML des URLs
        html1 = fetch_html(url1)
        html2 = fetch_html(url2)
        html3 = fetch_html(url3)
        
        # Extraction de texte
        text1 = extract_text_from_html(html1)
        text2 = extract_text_from_html(html2)
        text3 = extract_text_from_html(html3)
        
        # Fréquences de mots
        freq1 = get_word_frequencies(text1)
        freq2 = get_word_frequencies(text2)
        freq3 = get_word_frequencies(text3)
        
        # Génération de la liste sémantique complète
        ref_frequencies = normalize_frequencies([freq1, freq2, freq3])
        
        # Évaluation des sites
        score1 = evaluate_content(freq1, ref_frequencies)
        score2 = evaluate_content(freq2, ref_frequencies)
        score3 = evaluate_content(freq3, ref_frequencies)
        
        st.subheader("Résultats de l'analyse")
        st.write(f"\nScore du site 1 : {score1}/100")
        st.write(f"Score du site 2 : {score2}/100")
        st.write(f"Score du site 3 : {score3}/100")
        
        # Affichage des mots les plus pertinents
        st.subheader("Liste des mots à ajouter à votre article")
        df = pd.DataFrame(ref_frequencies.most_common(30), columns=["Mot", "Fréquence"])
        st.dataframe(df)
        
        # Export en CSV
        csv_file = "analyse_semantique.csv"
        df.to_csv(csv_file, index=False)
        with open(csv_file, "rb") as f:
            st.download_button("Télécharger le fichier CSV", f, file_name="analyse_semantique.csv", mime="text/csv")

if __name__ == "__main__":
    main()

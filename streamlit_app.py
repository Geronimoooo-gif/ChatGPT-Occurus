import streamlit as st
import pandas as pd
import re
import requests
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Configuration des poids SEO pour chaque balise HTML
TAG_WEIGHTS = {
    'title': 10.0,  # Plus haute importance
    'h1': 8.0,
    'h2': 6.0,
    'h3': 4.0,
    'h4': 3.0,
    'strong': 2.0,
    'a': 2.5,     # Les liens sont importants pour le SEO
    'p': 1.0,     # Texte normal
    'div': 0.8    # Contenu générique
}

class SEOAnalyzer:
    def __init__(self, keyword, urls):
        self.keyword = keyword
        self.urls = urls
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.stop_words = set(stopwords.words('french'))
        
    def extract_text_by_tags(self, html_content):
        """Extrait le texte en conservant l'information sur les balises"""
        soup = BeautifulSoup(html_content, "html.parser")
        tag_content = defaultdict(str)
        
        # Extraction du contenu pour chaque type de balise
        for tag_name in TAG_WEIGHTS.keys():
            elements = soup.find_all(tag_name)
            tag_content[tag_name] = ' '.join(elem.get_text(strip=True) for elem in elements)
            
        return tag_content
    
    def get_word_frequencies_by_tag(self, text_by_tags):
        """Calcule les fréquences des mots pour chaque type de balise"""
        frequencies_by_tag = {}
        
        for tag, content in text_by_tags.items():
            words = re.findall(r'\b\w{3,}\b', content.lower())
            words = [word for word in words if word not in self.stop_words]
            
            total_words = len(words) if words else 1
            frequencies = Counter(words)
            
            # Normalisation par 1000 mots et application du poids de la balise
            normalized_freq = {
                word: (count / total_words) * 1000 * TAG_WEIGHTS[tag]
                for word, count in frequencies.items()
            }
            frequencies_by_tag[tag] = Counter(normalized_freq)
            
        return frequencies_by_tag
    
    def calculate_weighted_score(self, frequencies_by_tag, reference_frequencies):
        """Calcule un score SEO pondéré basé sur l'emplacement des mots-clés"""
        total_score = 0
        max_possible_score = 0
        
        for word, ref_count in reference_frequencies.items():
            word_score = 0
            for tag, frequencies in frequencies_by_tag.items():
                if word in frequencies:
                    word_score += frequencies[word] * TAG_WEIGHTS[tag]
            
            # Score maximum possible pour ce mot
            max_word_score = ref_count * sum(TAG_WEIGHTS.values())
            
            total_score += min(word_score, max_word_score)
            max_possible_score += max_word_score
            
        return round((total_score / max_possible_score) * 100, 2) if max_possible_score > 0 else 0

    def analyze_url(self, url):
        """Analyse complète d'une URL"""
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            
            text_by_tags = self.extract_text_by_tags(response.text)
            frequencies_by_tag = self.get_word_frequencies_by_tag(text_by_tags)
            
            # Création d'un DataFrame pour les statistiques par balise
            tag_stats = []
            for tag, frequencies in frequencies_by_tag.items():
                for word, freq in frequencies.items():
                    tag_stats.append({
                        'url': url,
                        'tag': tag,
                        'word': word,
                        'frequency': freq,
                        'weighted_frequency': freq * TAG_WEIGHTS[tag]
                    })
            
            return frequencies_by_tag, pd.DataFrame(tag_stats)
            
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de l'analyse de {url}: {e}")
            return None, None

def main():
    st.title("Analyse Sémantique SEO Avancée")
    
    keyword = st.text_input("Mot-clé cible :", "")
    urls_input = st.text_area("URLs des sites concurrents (une par ligne)")
    
    if st.button("Analyser"):
        urls = list(filter(None, map(str.strip, urls_input.split("\n"))))
        
        if not keyword or len(urls) < 3:
            st.warning("Veuillez entrer un mot-clé et au moins 3 URLs.")
            return
        
        analyzer = SEOAnalyzer(keyword, urls)
        
        # Analyse de chaque URL
        all_frequencies = []
        all_tag_stats = []
        
        for url in urls:
            frequencies_by_tag, tag_stats = analyzer.analyze_url(url)
            if frequencies_by_tag and tag_stats is not None:
                all_frequencies.append(frequencies_by_tag)
                all_tag_stats.append(tag_stats)
        
        if all_tag_stats:
            # Combine tous les résultats
            combined_stats = pd.concat(all_tag_stats)
            
            # Création du tableau de bord des statistiques
            st.subheader("Analyse détaillée par balise HTML")
            
            # Tableau pivotant pour voir les statistiques par mot et par balise
            pivot_table = combined_stats.pivot_table(
                values='weighted_frequency',
                index=['word'],
                columns=['tag'],
                aggfunc='mean',
                fill_value=0
            ).round(2)
            
            st.dataframe(pivot_table)
            
            # Export des résultats
            csv_buffer = pivot_table.to_csv().encode()
            st.download_button(
                label="Télécharger l'analyse détaillée (CSV)",
                data=csv_buffer,
                file_name="analyse_seo_detaillee.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()

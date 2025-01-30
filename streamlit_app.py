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
    'title': 10.0,
    'h1': 8.0,
    'h2': 6.0,
    'h3': 4.0,
    'h4': 3.0,
    'strong': 2.0,
    'a': 2.5,
    'p': 1.0,
    'div': 0.8
}

ARTICLE_LENGTH = 1500

class SEOAnalyzer:
    def __init__(self, keyword, urls):
        self.keyword = keyword
        self.urls = urls
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('french'))
        
    def extract_text_by_tags(self, html_content):
        """Extrait le texte en conservant l'information sur les balises"""
        soup = BeautifulSoup(html_content, "html.parser")
        tag_content = defaultdict(str)
        
        for tag_name in TAG_WEIGHTS.keys():
            elements = soup.find_all(tag_name)
            tag_content[tag_name] = ' '.join(elem.get_text(strip=True) for elem in elements)
            
        full_text = ' '.join(soup.stripped_strings)
        return tag_content, full_text
    
    def get_word_frequencies_by_tag(self, text_by_tags):
        """Calcule les fréquences des mots pour chaque type de balise"""
        frequencies_by_tag = {}
        
        for tag, content in text_by_tags.items():
            words = re.findall(r'\b\w{3,}\b', content.lower())
            words = [word for word in words if word not in self.stop_words]
            
            total_words = len(words) if words else 1
            frequencies = Counter(words)
            
            normalized_freq = {
                word: (count / total_words) * 1000 * TAG_WEIGHTS[tag]
                for word, count in frequencies.items()
            }
            frequencies_by_tag[tag] = Counter(normalized_freq)
            
        return frequencies_by_tag

    def analyze_url(self, url):
        """Analyse complète d'une URL"""
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            
            text_by_tags, _ = self.extract_text_by_tags(response.text)
            frequencies_by_tag = self.get_word_frequencies_by_tag(text_by_tags)
            
            return frequencies_by_tag
            
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de l'analyse de {url}: {e}")
            return None

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
        
        all_tag_frequencies = []
        with st.spinner('Analyse en cours...'):
            for url in urls:
                result = analyzer.analyze_url(url)
                if result is not None:
                    all_tag_frequencies.append(result)

        # Vérification si nous avons des résultats à analyser
        if len(all_tag_frequencies) > 0:
            st.subheader("Suggestions d'optimisation SEO")
            combined_counts = defaultdict(lambda: defaultdict(int))
            
            for tag_frequencies in all_tag_frequencies:
                for tag, words in tag_frequencies.items():
                    for word in words:
                        combined_counts[word][tag] += 1
            
            suggestions = []
            for word, tags in combined_counts.items():
                for tag, count in tags.items():
                    if count >= len(urls) / 2:  # Si au moins 50% des concurrents l'utilisent
                        suggestions.append(f"Comme {count} des {len(urls)} acteurs les mieux positionnés, vous pourriez insérer l'occurrence \"{word}\" dans la balise {tag}.")
            
            if suggestions:
                st.write("\n".join(suggestions))
            else:
                st.write("Aucune suggestion d'optimisation basée sur vos concurrents.")
        else:
            st.error("Aucune donnée n'a pu être analysée. Veuillez vérifier les URLs et réessayer.")

if __name__ == "__main__":
    main()

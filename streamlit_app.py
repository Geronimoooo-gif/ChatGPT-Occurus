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
    
    def get_word_frequencies(self, text):
        words = re.findall(r'\b\w{3,}\b', text.lower())
        words = [word for word in words if word not in self.stop_words]
        total_words = len(words) if words else 1
        word_freq = Counter(words)
        normalized_freq = {word: (count / total_words) * 1000 for word, count in word_freq.items()}
        return Counter(normalized_freq), total_words
    
    def analyze_url(self, url):
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            
            text_by_tags, full_text = self.extract_text_by_tags(response.text)
            frequencies, total_words = self.get_word_frequencies(full_text)
            
            tag_counts = defaultdict(Counter)
            for tag, content in text_by_tags.items():
                words = re.findall(r'\b\w{3,}\b', content.lower())
                words = [word for word in words if word not in self.stop_words]
                tag_counts[tag] = Counter(words)
            
            return {'url': url, 'total_words': total_words, 'word_frequencies': frequencies, 'tag_counts': tag_counts}
        
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
        
        results = []
        with st.spinner('Analyse en cours...'):
            for url in urls:
                result = analyzer.analyze_url(url)
                if result is not None:
                    results.append(result)
        
        if not results:
            st.error("Aucune donnée n'a pu être analysée. Veuillez vérifier les URLs et réessayer.")
            return
        
        # Création des tableaux
        all_words = Counter()
        for res in results:
            all_words.update(res['word_frequencies'])
        
        df1 = pd.DataFrame(all_words.most_common(30), columns=["Mot", "Fréquence"])
        st.subheader("Analyse globale des mots-clés")
        st.dataframe(df1)
        
        # Tableau détaillé par balise
        all_tag_counts = defaultdict(lambda: defaultdict(int))
        for res in results:
            for tag, word_counts in res['tag_counts'].items():
                for word, count in word_counts.items():
                    all_tag_counts[word][tag] += count
        
        tag_table = pd.DataFrame.from_dict(all_tag_counts, orient='index').fillna(0).astype(int)
        st.subheader("Analyse détaillée par balise HTML")
        st.dataframe(tag_table)
        
        # Troisième tableau avec occurrences et densité
        detailed_data = []
        for res in results:
            url = res['url']
            total_words = res['total_words']
            for word, count in res['word_frequencies'].items():
                row = {
                    "URL": url,
                    "Mot clé": word,
                    "Total": count,
                    "Densité (%)": round((count / total_words) * 100, 2)
                }
                for tag in TAG_WEIGHTS.keys():
                    row[tag] = res['tag_counts'][tag].get(word, 0)
                detailed_data.append(row)
        
        df3 = pd.DataFrame(detailed_data)
        st.subheader("Détail des occurrences et densité par URL")
        st.dataframe(df3)

if __name__ == "__main__":
    main()

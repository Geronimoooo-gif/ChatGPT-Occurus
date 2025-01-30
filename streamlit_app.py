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
        """Calcule la fréquence des mots avec normalisation (pour l'ancien tableau)"""
        words = re.findall(r'\b\w{3,}\b', text.lower())
        words = [word for word in words if word not in self.stop_words]

        total_words = len(words) if words else 1
        word_freq = Counter(words)

        normalized_freq = {word: (count / total_words) * 1000 for word, count in word_freq.items()}
        return Counter(normalized_freq), total_words
    
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

    def filter_semantic_words(self, word_frequencies):
        """Filtre les mots selon leur similarité sémantique avec le mot-clé cible"""
        keyword_embedding = self.model.encode(self.keyword, convert_to_tensor=True)
        filtered_words = {}

        for word, freq in word_frequencies.items():
            word_embedding = self.model.encode(word, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(keyword_embedding, word_embedding).item()
            if similarity > 0.5:
                filtered_words[word] = freq

        return Counter(filtered_words)

    def calculate_recommended_frequencies(self, frequencies_list):
        """Calcule la fréquence médiane des mots et ajuste pour la longueur d'article cible"""
        word_occurrences = defaultdict(list)

        for freq_dict in frequencies_list:
            for word, freq in freq_dict.items():
                word_occurrences[word].append(freq)

        recommended_frequencies = {
            word: round(np.median(freq_list) * (ARTICLE_LENGTH / 1000))
            for word, freq_list in word_occurrences.items()
        }

        return Counter(recommended_frequencies)

    def analyze_url(self, url):
        """Analyse complète d'une URL"""
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            
            text_by_tags, full_text = self.extract_text_by_tags(response.text)
            frequencies_by_tag = self.get_word_frequencies_by_tag(text_by_tags)
            global_frequencies, total_words = self.get_word_frequencies(full_text)
            
            return {
                'tag_frequencies': frequencies_by_tag,
                'global_frequencies': global_frequencies,
                'total_words': total_words
            }
            
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
        
        all_global_frequencies = []
        all_total_words = {}
        
        with st.spinner('Analyse en cours...'):
            for url in urls:
                result = analyzer.analyze_url(url)
                if result is not None:
                    all_global_frequencies.append(result['global_frequencies'])
                    all_total_words[url] = result['total_words']
        
        if len(all_global_frequencies) > 0:
            raw_ref_frequencies = sum(all_global_frequencies, Counter())
            ref_frequencies = analyzer.filter_semantic_words(raw_ref_frequencies)
            recommended_freq = analyzer.calculate_recommended_frequencies(all_global_frequencies)
            
            st.subheader("Analyse globale des mots-clés")
            df1 = pd.DataFrame(ref_frequencies.most_common(30), columns=["Mot", "Fréquence"])
            df1["Fréquence conseillée"] = df1["Mot"].map(recommended_freq).fillna(0).astype(int)
            st.dataframe(df1)
            
            for word in df1["Mot"]:
                st.subheader(f"Analyse détaillée pour '{word}'")
                rows = []
                for url in urls:
                    count = all_global_frequencies[urls.index(url)].get(word, 0)
                    total_words = all_total_words.get(url, 1)
                    density = (count / total_words) * 100
                    rows.append([url, f"{total_words} mots", count, round(density, 2)])
                
                df_word = pd.DataFrame(rows, columns=["URL", "Total", "Occurrences", "Densité (%)"])
                st.dataframe(df_word)
        else:
            st.error("Aucune donnée analysée. Veuillez vérifier les URLs.")

if __name__ == "__main__":
    main()

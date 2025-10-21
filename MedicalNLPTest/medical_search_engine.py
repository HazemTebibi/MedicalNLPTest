# medical_search_engine_fixed.py
import streamlit as st
import pandas as pd
import pickle
import math
import os
import re
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Configuration de la page - SANS les options obsolètes
st.set_page_config(
    page_title="Moteur de Recherche Médicale",
    page_icon="🔍",
    layout="wide"
)

# Titre principal
st.title("🔍 Moteur de Recherche Médicale - Réanimation")
st.markdown("**Système d'indexation et de recherche dans un corpus médical spécialisé**")

# Initialisation de l'état de session
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'inverted_index' not in st.session_state:
    st.session_state.inverted_index = None
if 'search_engine' not in st.session_state:
    st.session_state.search_engine = None

class MedicalSearchEngine:
    def __init__(self, inverted_index):
        self.inverted_index = inverted_index
        self.doc_ids = self._get_all_doc_ids()
        self.N = len(self.doc_ids)
        self.avg_dl = self._calculate_avg_document_length()
        
    def _get_all_doc_ids(self):
        """Extrait tous les IDs de documents"""
        doc_ids = set()
        for term, docs in self.inverted_index.items():
            doc_ids.update(docs.keys())
        return list(doc_ids)
    
    def _calculate_avg_document_length(self):
        """Calcule la longueur moyenne des documents"""
        total_length = 0
        for doc_id in self.doc_ids:
            doc_length = 0
            for term_data in self.inverted_index.values():
                if doc_id in term_data:
                    doc_length += term_data[doc_id]['tf']
            total_length += doc_length
        return total_length / self.N if self.N > 0 else 0
    
    def preprocess_query(self, query):
        """Prétraite la requête utilisateur"""
        # Normalisation
        query = query.lower()
        query = unicodedata.normalize('NFKD', query)
        query = ''.join([c for c in query if not unicodedata.combining(c)])
        query = re.sub(r'[^a-zA-Z\s]', ' ', query)
        query = re.sub(r'\s+', ' ', query).strip()
        
        # Tokenisation
        tokens = word_tokenize(query, language='french')
        
        # Filtrage des stopwords
        french_stopwords = set(stopwords.words('french'))
        medical_whitelist = {'o2', 'fio2', 'ph', 'peep', 'pam', 'fc', 'fr', 'sao2'}
        french_stopwords = french_stopwords - medical_whitelist
        
        filtered_tokens = [
            token for token in tokens 
            if token not in french_stopwords and len(token) > 1
        ]
        return filtered_tokens
    
    def search(self, query, top_k=10, k1=1.2, b=0.75):
        """Recherche avec BM25"""
        query_terms = self.preprocess_query(query)
        
        if not query_terms:
            return [], query_terms
        
        scores = []
        
        for doc_id in self.doc_ids:
            score = 0
            doc_length = 0
            
            # Calculer la longueur du document
            for term_data in self.inverted_index.values():
                if doc_id in term_data:
                    doc_length += term_data[doc_id]['tf']
            
            for term in query_terms:
                if term in self.inverted_index and doc_id in self.inverted_index[term]:
                    tf = self.inverted_index[term][doc_id]['tf']
                    df = len(self.inverted_index[term])
                    idf = max(0, math.log((self.N - df + 0.5) / (df + 0.5) + 1))
                    
                    # Formule BM25
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_length / self.avg_dl))
                    
                    if denominator > 0:
                        score += idf * (numerator / denominator)
            
            if score > 0:
                scores.append((doc_id, score, doc_length))
        
        # Tri par score décroissant
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k], query_terms

def load_data():
    """Charge les données de recherche"""
    try:
        with open("inverted_index.pkl", "rb") as f:
            inverted_index = pickle.load(f)
        
        vocab_df = pd.read_csv("vocab.csv") if os.path.exists("vocab.csv") else None
        docs_df = pd.read_csv("docs.csv") if os.path.exists("docs.csv") else None
        
        search_engine = MedicalSearchEngine(inverted_index)
        return search_engine, vocab_df, docs_df, inverted_index
    
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {str(e)}")
        return None, None, None, None

# Sidebar pour le chargement
with st.sidebar:
    st.header("📂 Configuration")
    
    if st.button("🚀 Charger les données de recherche", use_container_width=True):
        with st.spinner("Chargement en cours..."):
            search_engine, vocab_df, docs_df, inverted_index = load_data()
            
            if search_engine is not None:
                st.session_state.data_loaded = True
                st.session_state.search_engine = search_engine
                st.session_state.vocab_df = vocab_df
                st.session_state.docs_df = docs_df
                st.session_state.inverted_index = inverted_index
                st.success("✅ Données chargées avec succès!")
            else:
                st.error("❌ Échec du chargement des données")

# Interface principale
if not st.session_state.data_loaded:
    st.warning("⚠️ Veuillez d'abord charger les données dans la sidebar")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📋 Fichiers requis:**
        - `inverted_index.pkl` - Index inversé
        - `vocab.csv` - Vocabulaire et pondérations  
        - `docs.csv` - Métadonnées des documents
        """)
    
    with col2:
        st.info("""
        **🎯 Instructions:**
        1. Cliquez sur *Charger les données*
        2. Attendez le message de confirmation
        3. Entrez votre requête médicale
        4. Consultez les résultats classés
        """)
    
    # Aperçu des fichiers disponibles
    st.subheader("📁 Fichiers disponibles")
    available_files = []
    for file in ['inverted_index.pkl', 'vocab.csv', 'docs.csv']:
        if os.path.exists(file):
            available_files.append(f"✅ {file}")
        else:
            available_files.append(f"❌ {file}")
    
    st.write("\n".join(available_files))

else:
    # Affichage des statistiques
    st.sidebar.header("📊 Statistiques")
    st.sidebar.metric("Documents indexés", st.session_state.search_engine.N)
    
    if st.session_state.vocab_df is not None:
        st.sidebar.metric("Termes uniques", len(st.session_state.vocab_df))
    
    st.sidebar.metric("Longueur moyenne", f"{st.session_state.search_engine.avg_dl:.0f} tokens")
    
    # Section de recherche
    st.header("🔎 Recherche médicale")
    
    query = st.text_input(
        "Entrez vos termes de recherche:",
        placeholder="ex: ventilation mécanique sepsis pression artérielle...",
        key="search_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col2:
        top_k = st.selectbox("Résultats par page", [5, 10, 20, 50], index=1)
    
    with col3:
        search_button = st.button("🔍 Lancer la recherche", use_container_width=True)
    
    # Exécution de la recherche
    if search_button or (query and st.session_state.get('last_query') != query):
        if query:
            st.session_state.last_query = query
            
            with st.spinner(f"Recherche de '{query}'..."):
                results, query_terms = st.session_state.search_engine.search(query, top_k=top_k)
                
                # Affichage des résultats
                st.subheader(f"📄 Résultats pour: '{query}'")
                
                if query_terms:
                    st.write(f"**Termes recherchés:** {', '.join(query_terms)}")
                
                if results:
                    st.write(f"**{len(results)} document(s) trouvé(s)**")
                    
                    for i, (doc_id, score, doc_length) in enumerate(results, 1):
                        with st.container():
                            st.markdown(f"### 📋 {doc_id} _(score: {score:.4f})_")
                            
                            # Métadonnées
                            col_meta1, col_meta2, col_meta3 = st.columns(3)
                            
                            with col_meta1:
                                st.metric("Score BM25", f"{score:.4f}")
                            
                            with col_meta2:
                                st.metric("Longueur", f"{doc_length} tokens")
                            
                            with col_meta3:
                                # Essayer de trouver des infos supplémentaires
                                if st.session_state.docs_df is not None:
                                    doc_info = st.session_state.docs_df[
                                        st.session_state.docs_df['doc_id'] == doc_id
                                    ]
                                    if not doc_info.empty:
                                        st.metric("Phrases", int(doc_info.iloc[0]['num_sentences']))
                            
                            # Extrait simulé
                            st.write("**Extrait:**")
                            st.write(f"*Document médical traitant de {', '.join(query_terms[:3])}. Contenu spécialisé en réanimation médicale avec des données cliniques détaillées...*")
                            
                            # Boutons d'action
                            col_btn1, col_btn2 = st.columns(2)
                            
                            with col_btn1:
                                if st.button(f"📖 Voir le document complet", key=f"view_{doc_id}"):
                                    st.info(f"Fonctionnalité d'affichage complet pour {doc_id} - À implémenter")
                            
                            with col_btn2:
                                if st.button(f"📊 Analyser les termes", key=f"analyze_{doc_id}"):
                                    # Afficher les termes correspondants
                                    matching_terms = []
                                    for term in query_terms:
                                        if term in st.session_state.inverted_index and doc_id in st.session_state.inverted_index[term]:
                                            tf = st.session_state.inverted_index[term][doc_id]['tf']
                                            matching_terms.append(f"{term} (tf={tf})")
                                    
                                    if matching_terms:
                                        st.success(f"**Termes correspondants:** {', '.join(matching_terms)}")
                            
                            st.markdown("---")
                else:
                    st.warning("Aucun document trouvé pour cette recherche. Essayez avec d'autres termes.")
        
        else:
            st.info("Veuillez entrer une requête de recherche.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**🔬 Projet TAL - Recherche Médicale**")
st.sidebar.markdown("*Moteur de recherche spécialisé en réanimation*")
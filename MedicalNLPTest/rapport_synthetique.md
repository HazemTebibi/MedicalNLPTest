
# RAPPORT SYNTHÉTIQUE - Moteur de Recherche Médicale

## Étapes réalisées
- Chargement et prétraitement du corpus médical
- Nettoyage linguistique complet (normalisation, tokenisation)
- Filtrage des stopwords avec liste blanche médicale
- Lemmatisation avec spaCy
- Construction du vocabulaire et index inversé
- Implémentation de l'algorithme BM25
- Interface Streamlit fonctionnelle

## Choix techniques
- Utilisation de spaCy pour la lemmatisation (meilleure précision)
- Liste blanche pour les termes médicaux courts mais importants
- BM25 pour le scoring (meilleur que TF-IDF pour la recherche)
- Stockage en pickle pour l'index (performance)

## Difficultés rencontrées
- Gestion des termes médicaux spécifiques
- Optimisation des performances pour les gros corpus
- Adaptation des stopwords au domaine médical

## Pistes d'amélioration
- Ajout de la recherche par phrases
- Interface avancée avec filtres
- Support des requêtes booléennes
- Visualisation des résultats

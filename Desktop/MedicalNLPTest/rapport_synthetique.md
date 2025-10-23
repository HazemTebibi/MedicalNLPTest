
# RAPPORT SYNTHÉTIQUE - Moteur de Recherche Médicale

## Étapes réalisées
- Chargement et prétraitement du corpus médical
- Nettoyage linguistique complet (normalisation, tokenisation)
- Lemmatisation avec spaCy
- Construction du vocabulaire et index inversé
- Implémentation de l'algorithme BM25
- Interface Streamlit fonctionnelle

## Choix techniques
- Utilisation de spaCy pour la lemmatisation (meilleure précision)
- Liste blanche pour les termes médicaux courts mais importants
- BM25 pour le scoring (meilleur que TF-IDF pour la recherche)


## Difficultés rencontrées
- Gestion des termes médicaux spécifiques


## Pistes d'amélioration
- Interface avancée avec filtres
- Support des requêtes booléennes
- Visualisation des résultats

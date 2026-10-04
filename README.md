# UC3 - Cardiac Telemonitoring

Projet de télésurveillance cardiaque du TP d'IA avancée. L'objectif est de
classifier des fenêtres ECG en cinq rythmes cardiaques et d'étudier le passage
des descripteurs du signal à une décision d'alerte.

## Objectifs

- charger et contrôler les ensembles ECG train, validation et test ;
- éviter les fuites entre ensembles en séparant les patients ;
- détecter les QRS et extraire des descripteurs temporels du rythme ;
- entraîner un modèle de classification reproductible ;
- évaluer les performances par classe, en particulier le rappel de la
	fibrillation atriale (FA) ;
- documenter les choix expérimentaux et les limites de la décision.

## État du projet

Le notebook `notebooks/uc3_jalon1.ipynb` contient l'implémentation du membre 1
(environnement, données, descripteurs, Q1 et vérification des poids de classe).
Les sections Q2, Q3 et Q4 sont encore réservées aux membres concernés.

Le modèle de référence est un MLP Keras :

```text
Dense(64, relu) -> Dropout(0,2) -> Dense(32, relu) -> Dense(5, softmax)
```

Il utilise Adam avec un taux d'apprentissage de `1e-3`, 60 époques, des lots de
64 et une entropie croisée pondérée par classe. La graine de reproductibilité
est fixée à `42`.

## Résultats disponibles

Les données actuellement présentes contiennent :

| Ensemble | Fenêtres | Points par fenêtre |
|---|---:|---:|
| Entraînement | 300 | 1 250 |
| Validation | 150 | 1 250 |
| Test | 1 050 | 1 250 |

Les cinq classes sont :

1. rythme sinusal ;
2. fibrillation atriale ;
3. extrasystoles ventriculaires ;
4. tachycardie sinusale ;
5. bloc AV du 2e degré.

Le modèle de référence atteint une accuracy de validation de `0,720`. Les
rappels observés sont respectivement `0,627`, `1,000`, `0,721`, `0,778` et
`0,880`. La précision de la FA reste faible malgré son rappel élevé, en raison
de nombreux faux positifs et du faible nombre d'exemples de FA en validation.

Le détecteur QRS signale 9 fenêtres en échec sur 150 selon les critères
documentés dans le notebook. Ces échecs concernent principalement la classe
bloc AV du 2e degré.

## Installation

Python 3.13 est utilisé pour les expériences documentées. La version minimale
exacte des bibliothèques n'est pas encore figée ; les dépendances principales
sont listées dans `requirements.txt`.

```bash
python3 -m pip install -r requirements.txt
```

## Exécution

Depuis la racine du projet, lancer Jupyter :

```bash
jupyter notebook notebooks/uc3_jalon1.ipynb
```

Le module commun peut être importé depuis un script ou une cellule située à la
racine du projet :

```python
from src.commun import charger, extraire, evaluer_rythme

train = charger("data", "train")
descripteurs = extraire(train["X"], age=train["age"], sexe=train["sexe"])
```

TensorFlow est importé uniquement au moment de construire le modèle avec
`mlp_rythme`, ce qui permet d'utiliser les fonctions de chargement et
d'extraction sans charger TensorFlow.

## Organisation

```text
.
├── data/                       # tableaux NumPy des splits train/val/test
├── docs/page_decisions.md      # décisions, seuil FA et vérifications théoriques
├── notebooks/uc3_jalon1.ipynb  # notebook principal du jalon 1
├── src/commun.py               # chargement, QRS, descripteurs et modèles
├── journal_experiences.csv     # journal des expériences
├── requirements.txt            # dépendances Python
└── README.md
```

## Méthode

Le signal est filtré dans la bande `5-15 Hz`. L'énergie de la dérivée au carré
est lissée sur 100 ms, puis les pics QRS sont détectés. Les 11 descripteurs
produits sont : nombre de QRS, moyenne, écart-type, minimum, maximum et médiane
des intervalles RR, RMSSD, pNN50, énergie du signal, âge et sexe.

Le `StandardScaler` doit être ajusté uniquement sur l'entraînement, puis appliqué
à la validation et au test. La sélection du modèle et du seuil d'alerte FA doit
être réalisée sur la validation ; le test ne doit être consulté qu'une seule
fois pour l'évaluation finale.

## Limites et précautions

- le détecteur QRS heuristique peut être sensible au bruit, à la dérive de
	ligne de base et aux morphologies atypiques ;
- les descripteurs agrégés ne conservent pas toute la forme temporelle de l'ECG ;
- la validation contient peu d'exemples de FA, ce qui rend les métriques
	sensibles à quelques erreurs ;
- l'accuracy seule ne suffit pas pour évaluer une alerte de télésurveillance ;
- ce projet expérimental ne constitue pas un dispositif médical ni une décision
	clinique autonome.

Les décisions expérimentales détaillées et les éléments restant à compléter
sont suivis dans [docs/page_decisions.md](docs/page_decisions.md).

# UC3 - Cardiac Telemonitoring (CardioPatch)

Projet de télésurveillance cardiaque du module d'IA Avancée (M2 SDIA, Année 2026–2027).  
L'objectif est d'analyser des fenêtres ECG de 10 secondes pour classifier automatiquement cinq rythmes cardiaques et concevoir un système d'alerte clinique pour la télésurveillance ambulatoire (patch connecté).

---

## 1. Objectifs du Jalon 1

- Charger et contrôler les ensembles ECG sans fuite patient (*patient leakage*) ;
- Détecter les complexes QRS et extraire des descripteurs physiologiques (temporels, morphologiques et spectraux) ;
- Entraîner et comparer des modèles de réseaux de neurones (MLP) sous différentes fonctions de perte (Entropie croisée simple, pondérée, Focal Loss) face au fort déséquilibre des classes ;
- Réaliser une étude d'ablation pour mesurer l'apport de chaque groupe de descripteurs ;
- Calibrer un seuil d'alerte clinique pour la fibrillation atriale (FA) garantissant au moins 95 % de spécificité (limitation de la fatigue d'alarme) ;
- Évaluer le système final **une seule fois** sur l'ensemble de test indépendant (1 050 fenêtres) ;
- Démontrer et valider les vérifications théoriques (poids de classes, gradient de l'entropie croisée).

---

## 2. Organisation de l'Équipe

Le travail du Jalon 1 est réparti entre les 3 membres de l'équipe :

| Membre | Responsabilités & Questions | Statut |
| :--- | :--- | :---: |
| **Membre 1** | Environnement, chargement des données, contrôles anti-fuite, détection QRS, 11 descripteurs de base, modèle MLP de référence (Q1), analyse des échecs QRS, vérification théorique 1 (poids de classe). | **Terminé & Validé** |
| **Membre 2** | Comparaison des fonctions de perte Q2 (CE simple, CE pondérée, Focal Loss $\gamma=2.0$), enrichissement à 24 descripteurs avancés (morphologie, Welch, Poincaré) et étude d'ablation Q3. | **Terminé & Validé** |
| **Membre 3** | Calibration du seuil d'alerte clinique FA Q4 (spécificité 95 %, courbe ROC / AUC), vérification théorique 2 (gradient de l'entropie croisée par rapport aux logits), tenue du journal d'expériences, évaluation finale sur le Test et rédaction de la note de décision clinique. | **Terminé & Validé** |

---

## 3. Données

Les données ECG sont échantillonnées à **125 Hz** (1 250 points par fenêtre de 10 secondes) issues d'enregistrements Holter :

| Ensemble | Fenêtres | Patients | Description |
| :--- | ---:| ---:| :--- |
| **Entraînement (*Train*)** | 300 | 12 | Apprentissage des poids du réseau et ajustement du `StandardScaler` |
| **Validation (*Val*)** | 150 | 6 | Sélection des hyperparamètres, étude d'ablation et calibration du seuil $s_{\text{FA}}$ |
| **Test (*Test*)** | 1 050 | 42 | Évaluation finale non biaisée, consultée **une seule fois** |

### Les 5 rythmes cardiaques :
0. **Rythme sinusal** (Normal)
1. **Fibrillation atriale** (FA / AFib - risque majeur d'AVC)
2. **Extrasystoles ventriculaires fréquentes** (EV / PVC)
3. **Tachycardie sinusale** (TS)
4. **Bloc Auriculo-Ventriculaire du 2e degré** (BAV2)

---

## 4. Méthodologie & Descripteurs

### Pipeline de Traitement du Signal
1. **Filtrage** : Passe-bande Butterworth 5–15 Hz (élimination des dérives respiratoires et bruits EMG).
2. **Détection QRS** : Énergie de la dérivée au carré (`gradient**2`), lissage par fenêtre glissante de 100 ms et détection de pics avec distance minimale de 250 ms.
3. **Normalisation** : `StandardScaler` ajusté **strictement sur l'entraînement**, puis appliqué à la validation et au test.

### Les 24 Descripteurs Physiologiques Extraits
* **Base temporelle (11 descripteurs)** : `n_qrs`, `rr_moy`, `rr_std`, `rr_min`, `rr_max`, `rr_med`, `rmssd`, `pnn50`, `energie`, `age`, `sexe`.
* **Morphologie QRS (3 descripteurs)** : `qrs_amp_moy`, `qrs_amp_std`, `qrs_larg_moy` (largeur à mi-hauteur via `scipy.signal.peak_widths`).
* **Analyse spectrale / Welch (5 descripteurs)** : puissances dans les bandes `p_vlf` (0.01–0.04 Hz), `p_lf` (0.04–0.15 Hz), `p_hf` (0.15–0.40 Hz), `ratio_lf_hf` et `entropie_spectrale`.
* **Irrégularité & Poincaré (5 descripteurs)** : coefficient de variation `rr_cv`, paramètres de l'ellipse `sd1`, `sd2`, `ratio_sd1_sd2` et `ratio_rmssd_sd`.

---

## 5. Synthèse des Résultats Expérimentaux

### 5.1 Comparatif des Modèles (Validation)

| Expérience | Descripteurs | Fonction de Perte | Val F1-Macro | Val F1 FA | Décision |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Q1. Baseline MLP (M1)** | Base (11) | CE Pondérée ($w_k$) | 0.6803 | 0.3158 | Modèle de référence initial |
| **Q2. Loss Comp. 1 (M2)** | Base (11) | CE Simple | 0.7367 | 0.3000 | Favorise la classe majoritaire |
| **Q2. Loss Comp. 2 (M2)** | Base (11) | Focal Loss ($\gamma=2.0$) | 0.6992 | 0.3243 | **Focal Loss retenue** (focus sur exemples durs) |
| **Q3. Ablation (+Freq) (M2)**| Base + Freq (16) | Focal Loss ($\gamma=2.0$) | **0.7323** | 0.3243 | Apport spectral de Welch validé |
| **Q4. Modèle Final (M3)** | Tous (24) | Focal Loss ($\gamma=2.0$) | 0.6272 | 0.3243 | Modèle complet déployé |

### 5.2 Seuil Clinique FA (Validation - M3)
- **Seuil calibré à 95 % de spécificité** : $s_{\text{FA}} = \mathbf{0{,}7026}$
- **Sensibilité FA en validation** : **83,33 %** (5/6 fenêtres)
- **Aire sous la courbe ROC (AUC)** : **0,9699**

### 5.3 Performances Finales sur l'Ensemble de Test (1 050 fenêtres - M3)

| Exigence Clinique | Mesure Obtenue | Cible Finale Cahier des Charges | Statut |
| :--- | :---: | :---: | :---: |
| **Reconnaissance des 5 rythmes** | F1 Macro = **0,740** (Accuracy 79,5 %) | $\ge 0,97$ | Étape Jalon 1 (MLP) |
| **Ne pas manquer une FA** (Sensibilité au seuil) | Sensibilité = **0,333** (Argmax = 0,465) | $\ge 0,95$ | En progression vers le CNN 1D |
| **Limiter les fausses alertes** (Sinus non alerté) | Part préservée = **0,995 (99,5 %)** | $\ge 0,97$ | **CIBLE DÉPASSÉE (+2,5 %)** |

---

## 6. Vérifications Théoriques

1. **Poids de classe ($w_k = \frac{N}{K \cdot N_k}$)** :
   Vérifié à la main et par le code pour $N=300$ et $K=5$ :
   $w_0 = 0{,}5042$, $w_1 = 0{,}8696$, $w_2 = 1{,}3043$, $w_3 = 2{,}0690$, $w_4 = 1{,}6216$. $\sum N_k w_k = 300 = N$.
2. **Gradient par rapport aux logits ($\nabla_z \mathcal{L} = \hat{p} - y$)** :
   Pour $\hat{p} = [0.50, 0.30, 0.10, 0.05, 0.05]$ et cible $y=1$ (FA) :
   - Perte d'entropie croisée : $\mathcal{L} = -\ln(0{,}30) \approx \mathbf{1{,}20397}$.
   - Gradient : $\nabla_z \mathcal{L} = \mathbf{[+0{,}50 ; -0{,}70 ; +0{,}10 ; +0{,}05 ; +0{,}05]}$, validé à $10^{-5}$ près par `tf.GradientTape()`.

---

## 7. Structure du Répertoire

```text
.
├── data/                       # Tableaux NumPy (.npy) des splits train/val/test
├── docs/
│   └── page_decisions.md       # Note de décision scientifique et clinique complète
├── notebooks/
│   ├── uc3_jalon1.ipynb        # Notebook principal complet et exécuté (M1 + M2 + M3)
│   └── journal_experiences.csv # Journal des expériences horodaté
├── src/
│   └── commun.py               # Utilitaires communs (QRS, 24 descripteurs, MLP, métriques)
├── journal_experiences.csv     # Journal des expériences à la racine
├── requirements.txt            # Dépendances du projet
└── README.md                   # Ce document
```

---

## 8. Installation et Exécution

### Installation des Dépendances
Python 3.11 ou 3.12/3.13 recommandé :
```bash
pip install -r requirements.txt
```

### Exécution du Notebook
Depuis la racine du projet :
```bash
jupyter notebook notebooks/uc3_jalon1.ipynb
```

Le module commun peut également être importé directement en Python :
```python
from src.commun import charger, extraire_avances, evaluer_rythme

train = charger("data", "train")
descripteurs_24 = extraire_avances(train["X"], age=train["age"], sexe=train["sexe"])
```

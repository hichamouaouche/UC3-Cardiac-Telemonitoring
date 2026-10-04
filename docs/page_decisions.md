# Page de Décisions - UC3 CardioPatch (Jalon 1)

**Projet** : CardioPatch - Télésurveillance par patch ECG  
**Jalon** : Jalon 1 - Des descripteurs à la décision  
**Auteurs** : Équipe CardioPatch (Membre 1, Membre 2, Membre 3)

---

## 1. Réponses aux Vérifications Théoriques

### 1.1 Calcul des Poids de Classe ($w_k = \frac{N}{K \cdot N_k}$) — [Membre 1]
La formule de pondération rééquilibrant les classes déséquilibrées est :
$$w_k = \frac{N}{K \cdot N_k}$$
où $N = 300$ (effectif total de l'ensemble d'entraînement Holter), $K = 5$ (nombre de classes de rythmes cardiaques), et $N_k$ est l'effectif de la classe $k$.

| Classe $k$ | Rythme Cardiaque | Effectif ($N_k$) | Formule $\frac{300}{5 \cdot N_k}$ | Poids Théorique $w_k$ |
| :---: | :--- | :---: | :---: | :---: |
| **0** | Rythme sinusal | 119 | $\frac{300}{595}$ | **0.5042** |
| **1** | Fibrillation atriale (FA) | 69 | $\frac{300}{345}$ | **0.8696** |
| **2** | Extrasystoles ventriculaires fréquentes (EV) | 46 | $\frac{300}{230}$ | **1.3043** |
| **3** | Tachycardie sinusale (TS) | 29 | $\frac{300}{145}$ | **2.0690** |
| **4** | Bloc AV du 2e degré (BAV2) | 37 | $\frac{300}{185}$ | **1.6216** |

**Propriété vérifiée** : $\sum_{k=0}^4 N_k w_k = 300 = N$. Le calcul manuel, la fonction `poids_classes()` et `scikit-learn.compute_class_weight` concordent parfaitement.

---

### 1.2 Perte d'Entropie Croisée et Gradient par rapport aux Logits — [Membre 3]
Soit une fenêtre de **fibrillation atriale (classe 1)** avec vecteur cible *one-hot* $y = [0, 1, 0, 0, 0]$, et probabilités prédites Softmax :
$$\hat{p} = [0{,}50 ;\\; 0{,}30 ;\\; 0{,}10 ;\\; 0{,}05 ;\\; 0{,}05]$$

1. **Perte d'Entropie Croisée $\mathcal{L}$** :
   $$\mathcal{L} = -\sum_{k=0}^{4} y_k \ln(\hat{p}_k) = -\ln(\hat{p}_1) = -\ln(0{,}30) \approx \mathbf{1{,}20397}$$

2. **Gradient par rapport aux logits $z$ ($\nabla_z \mathcal{L}$)** :
   Comme $\hat{p}_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$, la dérivée partielle s'écrit :
   $$\frac{\partial \mathcal{L}}{\partial z_k} = \hat{p}_k - y_k \implies \nabla_z \mathcal{L} = \hat{p} - y = \mathbf{[0{,}50 ;\\; -0{,}70 ;\\; 0{,}10 ;\\; 0{,}05 ;\\; 0{,}05]}$$

**Interprétation dynamique** : Le gradient pour la classe 1 est négatif ($-0{,}70$), ce qui augmente son logit $z_1$ lors de la descente de gradient, tendant à corriger la sous-estimation de la FA. Pour les autres classes, le gradient est positif, ce qui réduit leurs logits.

---

## 2. Synthèse des Expériences et Choix Retenus

| Expérience | Descripteurs | Perte | Val F1-Macro | Val F1 FA | Test F1-Macro | Décision / Statut |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Q1. Baseline MLP (M1)** | Base (11) | CE Pondérée ($w_k$) | 0.6803 | 0.3158 | - | Modèle de référence initial |
| **Q2. Loss Comp. 1 (M2)** | Base (11) | CE Simple | 0.7367 | 0.3000 | - | Favorise les classes majoritaires |
| **Q2. Loss Comp. 2 (M2)** | Base (11) | Focal Loss ($\gamma=2.0$) | 0.6992 | 0.3243 | - | Meilleur focus sur les exemples durs |
| **Q3. Ablation (+Freq) (M2)** | Base + Freq (16) | Focal Loss ($\gamma=2.0$) | **0.7323** | 0.3243 | - | Apport spectral de Welch validé |
| **Q4. Modèle Final (M3)** | Tous (24) | Focal Loss ($\gamma=2.0$) | 0.6272 | 0.3243 | **0.7395** | Modèle complet déployé sur Test |

---

## 3. Réglage du Seuil d'Alerte FA & Analyse Clinique — [Membre 3]

### 3.1 Seuil d'Alerte pour 95 % de Spécificité FA
- **Calibré sur l'ensemble de validation** (sur les fenêtres non-FA) : $s_{\text{FA}} = \mathbf{0{,}7026}$.
- **Aire sous la courbe ROC (AUC)** : **0,9699** (excellente séparabilité globale des scores).
- **Sensibilité observée sur la validation** à ce seuil : **83,33 %** (5 fenêtres FA détectées sur 6).

### 3.2 Performances sur l'Ensemble de Test (1 050 fenêtres, 42 patients indépendants)

| Exigence Clinique | Mesure Test Obtenue | Cible Finale Cahier des Charges | Écart |
| :--- | :---: | :---: | :---: |
| **Reconnaître les 5 rythmes** | F1 Macro = **0,740** (Accuracy 79,5 %) | $\ge 0,97$ | -0,230 |
| **Ne pas manquer une FA** | Sensibilité au seuil = **0,333** (Argmax = 0,465) | $\ge 0,95$ | -0,617 |
| **Limiter les fausses alertes** | Part des sinus non alertés = **0,995 (99,5 %)** | $\ge 0,97$ | **+0,025 (Cible dépassée !)** |

### 3.3 Analyse Clinique et Compromis Faux Positifs vs Faux Négatifs
1. **Risque Vital (Faux Négatif de FA)** :
   Une crise de fibrillation atriale non diagnostiquée expose le patient à une stagnation sanguine intracardiaque et à la survenue d'un **AVC ischémique massif**.
2. **Fatigue d'Alarme (Faux Positif)** :
   Des alertes intempestives sur rythme sinusal saturent les lignes de télésurveillance et conduisent les soignants à désactiver ou ignorer les alarmes.
3. **Bilan du Jalon 1** :
   Avec un seuil de spécificité à 95 %, **99,5 % des rythmes sinusaux sont parfaitement préservés de toute alerte intempestive**. La sensibilité sur le test reste toutefois insuffisante à ce stade, ce qui s'explique par la nature simplifiée des descripteurs scalaires d'un MLP. Les jalons suivants (réseaux convolutifs 1D / CNN) permettront d'extraire la morphologie fine de l'onde P pour atteindre les $\ge 95\%$ de sensibilité exigés.

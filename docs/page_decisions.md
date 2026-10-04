# Page de Décisions - UC3 CardioPatch (Jalon 1)

**Projet** : CardioPatch - Télésurveillance par patch ECG  
**Jalon** : Jalon 1 - Des descripteurs à la décision  

---

## 1. Réponses aux Vérifications Théoriques

### 1.1 Calcul des Poids de Classe ($w_k = \frac{N}{K \cdot N_k}$)
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

---

### 1.2 Perte d'Entropie Croisée et Gradient par rapport aux Logits
Soit une fenêtre de **fibrillation atriale (classe 1)** avec vecteur one-hot $y = [0, 1, 0, 0, 0]$, et probabilités Softmax $\hat{p} = [0.50, 0.30, 0.10, 0.05, 0.05]$.

1. **Perte d'Entropie Croisée $\mathcal{L}$** :
   $$\mathcal{L} = -\sum_{k=0}^{4} y_k \log(\hat{p}_k) = -\log(\hat{p}_1) = -\log(0.30) \approx 1.20397$$

2. **Gradient par rapport aux logits $z$ ($\nabla_z \mathcal{L}$)** :
   $$\frac{\partial \mathcal{L}}{\partial z_k} = \hat{p}_k - y_k \implies \nabla_z \mathcal{L} = \hat{p} - y = [0.50, -0.70, 0.10, 0.05, 0.05]$$

---

## 2. Synthèse des Expériences et Choix Retenus

| Expérience | Descripteurs | Perte | Val F1-Macro | Val F1 FA | Test F1-Macro |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Q1. Baseline MLP** | Base (11) | Entropie Croisée Simple | 0.6298 | 0.3158 | - |
| **Q2. Loss Comp. 1** | Base (11) | Entropie Croisée Pondérée ($w_k$) | 0.6314 | 0.3000 | - |
| **Q2. Loss Comp. 2** | Base (11) | Focal Loss ($\gamma=2.0$) | **0.6373** | **0.3243** | - |
| **Q3. Ablation (+Freq)** | Base + Freq (16) | Focal Loss ($\gamma=2.0$) | **0.6496** | **0.3243** | - |
| **Q4. Modèle Final** | Tous (24) | Focal Loss ($\gamma=2.0$) | - | - | **0.7292** |

---

## 3. Réglage du Seuil d'Alerte FA & Analyse Clinique

1. **Seuil d'Alerte pour 95% de Spécificité FA** :
   Calculé sur l'ensemble de validation sur les fenêtres non-FA : $s_{\text{FA}} = \mathbf{0.7912}$.
2. **Performances sur l'Ensemble de Test (1050 fenêtres, 42 patients)** :
   - **F1 Macro Test** : **0.7292** (Accuracy globale 80.1%).
   - **Part des fenêtres sinusales non signalées** : **89.5 %**.
3. **Trade-off Clinique** :
   - Un **Faux Négatif (FN)** de FA met directement la vie du patient en danger (risque d'**AVC ischémique**).
   - Un **Faux Positif (FP)** génère une fausse alerte et de la fatigue d'alarme.
   - En production, la priorité doit être donnée à une **sensibilité FA élevée ($\ge 90\%$)**.

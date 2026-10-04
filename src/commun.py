"""Utilitaires communs pour le jalon 1 CardioPatch.

Le module ne charge pas TensorFlow a l'import afin de rester utilisable pour
l'exploration des donnees et les tests legers.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt, welch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
FS = 125
NOMS = (
    "sinusal",
    "fibrillation atriale",
    "extrasystoles ventriculaires",
    "tachycardie sinusale",
    "bloc AV du 2e degre",
)
NOMS_FEAT = (
    "n_qrs",
    "rr_moy",
    "rr_std",
    "rr_min",
    "rr_max",
    "rr_med",
    "rmssd",
    "pnn50",
    "energie",
    "age",
    "sexe",
)

NOMS_FEAT_AVANCES = NOMS_FEAT + (
    "qrs_amp_moy",
    "qrs_amp_std",
    "qrs_larg_moy",
    "p_vlf",
    "p_lf",
    "p_hf",
    "ratio_lf_hf",
    "entropie_spectrale",
    "rr_cv",
    "sd1",
    "sd2",
    "ratio_sd1_sd2",
    "ratio_rmssd_sd",
)


def _find_array(data_dir: Path, prefix: str, split: str) -> Path:
    """Trouve un fichier prefixe_split.npy en acceptant aussi split_prefix.npy."""
    candidates = (data_dir / f"{prefix}_{split}.npy", data_dir / f"{split}_{prefix}.npy")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Fichier absent pour {prefix}/{split}: {candidates[0]}")


def charger(data_dir: str | Path, split: str) -> dict[str, np.ndarray]:
    """Charge les six tableaux d'un split (train, val ou test)."""
    data_path = Path(data_dir)
    prefixes = {"X": "X", "y": "classe", "fc": "fc", "patient": "patient", "age": "age", "sexe": "sexe"}
    return {name: np.load(_find_array(data_path, prefix, split), allow_pickle=False) for name, prefix in prefixes.items()}


def verifier_sans_fuite(*splits: dict[str, np.ndarray]) -> bool:
    """Verifie que les ensembles fournis n'ont aucun patient en commun."""
    patient_sets = [set(np.asarray(split["patient"]).reshape(-1).tolist()) for split in splits]
    for index, current in enumerate(patient_sets):
        for other in patient_sets[index + 1 :]:
            if current & other:
                raise ValueError("Fuite de patients detectee entre deux splits")
    return True


def poids_classes(y: np.ndarray, n_classes: int = len(NOMS)) -> dict[int, float]:
    """Calcule les poids balanced de chaque classe, y compris les classes absentes."""
    labels = np.asarray(y).reshape(-1).astype(int)
    present = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=present, y=labels)
    result = {class_id: 1.0 for class_id in range(n_classes)}
    result.update({int(class_id): float(weight) for class_id, weight in zip(present, weights)})
    return result


def detecter_qrs(signal_in: np.ndarray, fs: int = FS) -> np.ndarray:
    """Detecte les QRS par bande 5-15 Hz, energie et pics espaces."""
    values = np.asarray(signal_in, dtype=float).reshape(-1)
    if values.size < 16:
        return np.array([], dtype=int)
    low = 5.0 / (fs / 2.0)
    high = min(15.0 / (fs / 2.0), 0.99)
    sos = butter(3, [low, high], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, values - np.median(values))
    energy = uniform_filter1d(np.gradient(filtered) ** 2, size=max(1, int(0.1 * fs)))
    distance = max(1, int(0.25 * fs))
    prominence = max(float(np.std(energy) * 0.25), np.finfo(float).eps)
    height = float(np.median(energy) + 0.5 * np.std(energy))
    peaks, _ = find_peaks(energy, distance=distance, height=height, prominence=prominence)
    return peaks.astype(int)


def _encoder_sexe(value: Any) -> float:
    """Encode le sexe numerique ou textuel sans supposer un format unique."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0 if str(value).strip().lower() in {"m", "male", "homme", "1"} else 0.0


def descripteurs_base(signal_in: np.ndarray, fs: int = FS, age: Any = 0.0, sexe: Any = 0.0) -> np.ndarray:
    """Construit les 11 descripteurs QRS, rythme et variables patient."""
    values = np.asarray(signal_in, dtype=float).reshape(-1)
    qrs = detecter_qrs(values, fs)
    rr = np.diff(qrs) / fs if qrs.size > 1 else np.array([], dtype=float)
    successive_diff = np.diff(rr) if rr.size > 1 else np.array([], dtype=float)
    energy = np.mean(np.gradient(values) ** 2) if values.size else 0.0
    return np.array(
        [
            float(qrs.size),
            float(np.mean(rr)) if rr.size else 0.0,
            float(np.std(rr)) if rr.size else 0.0,
            float(np.min(rr)) if rr.size else 0.0,
            float(np.max(rr)) if rr.size else 0.0,
            float(np.median(rr)) if rr.size else 0.0,
            float(np.sqrt(np.mean(successive_diff**2))) if successive_diff.size else 0.0,
            float(np.mean(np.abs(successive_diff) > 0.05)) if successive_diff.size else 0.0,
            float(energy),
            float(np.asarray(age).reshape(-1)[0]) if np.asarray(age).size else 0.0,
            _encoder_sexe(np.asarray(sexe).reshape(-1)[0]) if np.asarray(sexe).size else 0.0,
        ],
        dtype=float,
    )


def extraire(X: np.ndarray, fs: int = FS, age: np.ndarray | None = None, sexe: np.ndarray | None = None) -> np.ndarray:
    """Extrait les 11 descripteurs pour chaque fenetre ECG."""
    if not np.isscalar(fs):
        fs, age, sexe = sexe if np.isscalar(sexe) else FS, fs, age
    windows = np.asarray(X)
    if windows.ndim == 1:
        windows = windows[None, :]
    ages = np.zeros(len(windows)) if age is None else np.asarray(age).reshape(-1)
    sexes = np.zeros(len(windows)) if sexe is None else np.asarray(sexe).reshape(-1)
    if len(ages) != len(windows) or len(sexes) != len(windows):
        raise ValueError("age et sexe doivent contenir une valeur par fenetre")
    return np.vstack([descripteurs_base(window, fs=fs, age=ages[index], sexe=sexes[index]) for index, window in enumerate(windows)])


# --- FONCTIONS AJOUTÉES PAR LE MEMBRE 2 (Descripteurs Avancés & Ablation) ---

def descripteurs_avances(signal_in: np.ndarray, fs: int = FS, age: Any = 0.0, sexe: Any = 0.0) -> np.ndarray:
    """Construit les 24 descripteurs complets (Base + Morphologie + Fréquence + Irrégularité)."""
    base_f = descripteurs_base(signal_in, fs=fs, age=age, sexe=sexe)
    values = np.asarray(signal_in, dtype=float).reshape(-1)
    qrs = detecter_qrs(values, fs)
    rr = np.diff(qrs) / fs if qrs.size > 1 else np.array([], dtype=float)
    
    # 1. Morphologie QRS
    if qrs.size > 0:
        amps = values[qrs]
        amp_moy = float(np.mean(amps))
        amp_std = float(np.std(amps)) if len(amps) > 1 else 0.0
        widths = signal.peak_widths(values, qrs, rel_height=0.5)[0] / fs
        larg_moy = float(np.mean(widths))
    else:
        amp_moy, amp_std, larg_moy = 0.0, 0.0, 0.1
    morph_f = [amp_moy, amp_std, larg_moy]
    
    # 2. Puissance Spectrale (Welch)
    freqs, psd = welch(values, fs=fs, nperseg=min(256, len(values)))
    psd_norm = psd / (np.sum(psd) + 1e-8)
    spec_entropy = float(-np.sum(psd_norm * np.log(psd_norm + 1e-8)))
    
    vlf_mask = (freqs >= 0.01) & (freqs < 0.04)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)
    
    p_vlf = float(np.sum(psd[vlf_mask]))
    p_lf = float(np.sum(psd[lf_mask]))
    p_hf = float(np.sum(psd[hf_mask]))
    ratio_lf_hf = float(p_lf / (p_hf + 1e-8))
    freq_f = [p_vlf, p_lf, p_hf, ratio_lf_hf, spec_entropy]
    
    # 3. Irrégularité & Poincaré
    rr_cv = float(base_f[2] / (base_f[1] + 1e-8)) if base_f[1] > 0 else 0.0
    if len(rr) > 1:
        diff_rr = np.diff(rr)
        sd1 = float(np.sqrt(0.5 * np.var(diff_rr)))
        sd2 = float(np.sqrt(max(0, 2 * np.var(rr) - 0.5 * np.var(diff_rr))))
        ratio_sd1_sd2 = float(sd1 / (sd2 + 1e-8))
        ratio_rmssd_sd = float(base_f[6] / (base_f[2] + 1e-8))
    else:
        sd1, sd2, ratio_sd1_sd2, ratio_rmssd_sd = 0.0, 0.0, 0.0, 0.0
    irreg_f = [rr_cv, sd1, sd2, ratio_sd1_sd2, ratio_rmssd_sd]
    
    return np.concatenate([base_f, morph_f, freq_f, irreg_f])


def extraire_avances(X: np.ndarray, fs: int = FS, age: np.ndarray | None = None, sexe: np.ndarray | None = None) -> np.ndarray:
    """Extrait les 24 descripteurs pour chaque fenetre ECG."""
    windows = np.asarray(X)
    if windows.ndim == 1:
        windows = windows[None, :]
    ages = np.zeros(len(windows)) if age is None else np.asarray(age).reshape(-1)
    sexes = np.zeros(len(windows)) if sexe is None else np.asarray(sexe).reshape(-1)
    return np.vstack([descripteurs_avances(window, fs=fs, age=ages[index], sexe=sexes[index]) for index, window in enumerate(windows)])


# --- FONCTIONS AJOUTÉES PAR LE MEMBRE 3 (Seuil d'Alerte & Évaluation) ---

def seuil_specificite(y_true: np.ndarray, score_fa: np.ndarray, specificite: float = 0.95) -> float:
    """Calcule le seuil d'alerte FA pour une specificite donnee sur les fenetres non-FA."""
    non_fa_scores = score_fa[np.asarray(y_true) != 1]
    return float(np.quantile(non_fa_scores, specificite))


def mlp_rythme(input_dim: int, n_classes: int = len(NOMS), loss: str = "sparse_categorical_crossentropy", **kwargs: Any) -> Any:
    """Construit et compile un MLP Keras reproductible pour les cinq rythmes."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    optimizer = kwargs.pop("optimizer", tf.keras.optimizers.Adam(learning_rate=1e-3))
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


def evaluer_rythme(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Retourne accuracy, matrice de confusion et rapport par classe."""
    truth = np.asarray(y_true).reshape(-1).astype(int)
    predictions = np.asarray(y_pred)
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "matrice_confusion": confusion_matrix(truth, predictions, labels=np.arange(len(NOMS))),
        "rapport": classification_report(truth, predictions, labels=np.arange(len(NOMS)), target_names=NOMS, zero_division=0, output_dict=True),
    }


def journaliser(membre: int | str, question: str, configuration: str, resultat: str, decision: str, path: str | Path = "journal_experiences.csv") -> None:
    """Ajoute un essai au journal CSV sans ecraser les experiences precedentes."""
    journal_path = Path(path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow([datetime.now(timezone.utc).isoformat(), membre, question, configuration, resultat, decision])

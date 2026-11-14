"""
entrenar_baseline_fft_logreg_v2.py
===============================

Baseline rápido e interpretable para OSDB:
- Entrada: bloques de 5 s, 125 muestras de magnitud a 25 Hz.
- Extrae características temporales y espectrales.
- Entrena regresión logística con escalado.
- Balancea por EVENTO: cada evento aporta el mismo peso total.
- Selecciona C y umbral usando SOLO validación.
- NO utiliza testData.csv.
- Guarda modelo, umbrales, métricas, probabilidades y gráficas.

Por qué:
El primer DeepEpi de 3 épocas produjo probabilidades casi constantes.
Este baseline permite comprobar si las señales/etiquetas contienen
separación antes de invertir más tiempo en una CNN profunda.

Ejecución:
    python .\entrenar_baseline_fft_logreg_v2.py
"""

from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import math
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import get_window
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# RUTAS
# ============================================================

PIPELINE_DIR = Path(r"C:\Users\cesar\osd\pipeline_prototipo_v2")
TRAIN_PATH = PIPELINE_DIR / "trainData.csv"
VAL_PATH = PIPELINE_DIR / "valData.csv"

OUTPUT_DIR = PIPELINE_DIR / "baseline_fft_logreg_v2"

FS_HZ = 25.0
N_SAMPLES = 125

# Candidatos de regularización.
C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


# ============================================================
# UTILIDADES
# ============================================================

def normalize_label(value: Any) -> int:
    if isinstance(value, str):
        text = value.strip().lower()

        if text in {"1", "1.0", "seizure"}:
            return 1

        if text in {
            "0",
            "0.0",
            "false alarm",
            "nda",
            "non-seizure",
            "nonseizure",
        }:
            return 0

    number = float(value)

    if number == 1:
        return 1

    if number == 0:
        return 0

    raise ValueError(f"Etiqueta no binaria: {value!r}")


def magnitude_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        f"M{i:03d}"
        for i in range(N_SAMPLES)
    ]

    missing = [
        col
        for col in cols
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Faltan columnas de magnitud: "
            + ", ".join(missing[:20])
        )

    return cols


def band_power(
    power: np.ndarray,
    freqs: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    """
    Potencia sumada en [low_hz, high_hz).
    """
    mask = (
        (freqs >= low_hz)
        & (freqs < high_hz)
    )

    if not np.any(mask):
        return 0.0

    return float(power[mask].sum())


def spectral_entropy(power: np.ndarray) -> float:
    values = power.astype(np.float64, copy=True)
    total = values.sum()

    if total <= 0:
        return 0.0

    probabilities = values / total
    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    maximum = math.log2(len(values))

    if maximum <= 0:
        return 0.0

    return float(entropy / maximum)


def extract_one_window_features(
    magnitude_mg: np.ndarray,
) -> dict[str, float]:
    """
    Convierte 125 muestras de magnitud en características.

    OSDB usa aproximadamente 1000 mg en reposo.
    Se trabaja en g para que los valores sean comparables con el MPU6050.
    """
    magnitude_g = (
        np.asarray(magnitude_mg, dtype=np.float64)
        / 1000.0
    )

    if magnitude_g.size != N_SAMPLES:
        raise ValueError(
            f"Se esperaban {N_SAMPLES} muestras; "
            f"llegaron {magnitude_g.size}."
        )

    if not np.isfinite(magnitude_g).all():
        raise ValueError(
            "La ventana contiene muestras no finitas."
        )

    # Componente dinámica: se elimina el nivel DC/gravedad medio.
    dynamic = magnitude_g - np.mean(magnitude_g)

    # Ventana Hann para FFT.
    hann = get_window("hann", N_SAMPLES, fftbins=True)
    spectrum = np.fft.rfft(dynamic * hann)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(
        N_SAMPLES,
        d=1.0 / FS_HZ,
    )

    # Eliminar DC del cálculo espectral.
    non_dc = freqs > 0
    power_non_dc = power[non_dc]
    freqs_non_dc = freqs[non_dc]

    total_power = band_power(
        power,
        freqs,
        0.2,
        FS_HZ / 2 + 1e-9,
    )

    power_0_1 = band_power(power, freqs, 0.2, 1.0)
    power_1_3 = band_power(power, freqs, 1.0, 3.0)
    power_3_8 = band_power(power, freqs, 3.0, 8.0)
    power_5_10 = band_power(power, freqs, 5.0, 10.0)
    power_8_12_5 = band_power(
        power,
        freqs,
        8.0,
        FS_HZ / 2 + 1e-9,
    )

    epsilon = 1e-12

    if power_non_dc.size:
        dominant_index = int(
            np.argmax(power_non_dc)
        )
        dominant_frequency = float(
            freqs_non_dc[dominant_index]
        )
        dominant_power = float(
            power_non_dc[dominant_index]
        )
    else:
        dominant_frequency = 0.0
        dominant_power = 0.0

    diff = np.diff(magnitude_g)
    jerk = diff * FS_HZ

    mean_g = float(np.mean(magnitude_g))
    std_g = float(np.std(magnitude_g))
    rms_dynamic_g = float(
        np.sqrt(np.mean(dynamic**2))
    )
    mean_abs_dynamic_g = float(
        np.mean(np.abs(dynamic))
    )
    max_abs_dynamic_g = float(
        np.max(np.abs(dynamic))
    )
    peak_to_peak_g = float(
        np.ptp(magnitude_g)
    )
    jerk_rms_g_s = float(
        np.sqrt(np.mean(jerk**2))
    )

    # Autocorrelación normalizada: máximo entre 0.2 y 2 s.
    dynamic_energy = float(
        np.dot(dynamic, dynamic)
    )

    if dynamic_energy > epsilon:
        autocorr = np.correlate(
            dynamic,
            dynamic,
            mode="full",
        )[N_SAMPLES - 1 :]

        autocorr = autocorr / autocorr[0]

        lag_min = max(1, int(round(0.2 * FS_HZ)))
        lag_max = min(
            len(autocorr),
            int(round(2.0 * FS_HZ)) + 1,
        )

        autocorr_peak = float(
            np.max(autocorr[lag_min:lag_max])
        )
    else:
        autocorr_peak = 0.0

    return {
        "mean_g": mean_g,
        "std_g": std_g,
        "rms_dynamic_g": rms_dynamic_g,
        "mean_abs_dynamic_g": mean_abs_dynamic_g,
        "max_abs_dynamic_g": max_abs_dynamic_g,
        "peak_to_peak_g": peak_to_peak_g,
        "jerk_rms_g_s": jerk_rms_g_s,
        "autocorr_peak_0_2_2s": autocorr_peak,
        "total_power_0_2_12_5": total_power,
        "power_0_2_1": power_0_1,
        "power_1_3": power_1_3,
        "power_3_8": power_3_8,
        "power_5_10": power_5_10,
        "power_8_12_5": power_8_12_5,
        "ratio_3_8": power_3_8 / (total_power + epsilon),
        "ratio_5_10": power_5_10 / (total_power + epsilon),
        "dominant_frequency_hz": dominant_frequency,
        "dominant_power_ratio": (
            dominant_power / (total_power + epsilon)
        ),
        "spectral_entropy": spectral_entropy(
            power_non_dc
        ),
    }


def create_feature_table(
    source_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    cols = magnitude_columns(source_df)

    rows: list[dict[str, Any]] = []

    for row_index, row in source_df.iterrows():
        values = row[cols].to_numpy(
            dtype=np.float64,
        )

        features = extract_one_window_features(
            values
        )

        output = {
            "source_row": int(row_index),
            "eventId": row["eventId"],
            "userId": row["userId"],
            "typeStr": row["typeStr"],
            "label": normalize_label(row["type"]),
        }

        output.update(features)
        rows.append(output)

    feature_df = pd.DataFrame(rows)

    feature_names = [
        column
        for column in feature_df.columns
        if column not in {
            "source_row",
            "eventId",
            "userId",
            "typeStr",
            "label",
        }
    ]

    return feature_df, feature_names


def event_balanced_weights(
    feature_df: pd.DataFrame,
) -> np.ndarray:
    """
    Cada evento aporta el mismo peso total.
    Como train tiene igual cantidad de eventos positivos y negativos,
    esto también evita que eventos largos dominen.
    """
    event_sizes = (
        feature_df.groupby("eventId")["eventId"]
        .transform("size")
        .to_numpy(dtype=np.float64)
    )

    weights = 1.0 / event_sizes

    # Normalizar para que el peso medio sea 1.
    weights = weights / np.mean(weights)

    return weights


def metrics_at_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    fpr = 1.0 - specificity
    balanced_accuracy = balanced_accuracy_score(
        y_true,
        predictions,
    )

    f1 = (
        2 * precision * sensitivity
        / (precision + sensitivity)
        if precision + sensitivity > 0
        else 0.0
    )

    beta = 2.0
    f2 = (
        (1 + beta**2)
        * precision
        * sensitivity
        / (beta**2 * precision + sensitivity)
        if beta**2 * precision + sensitivity > 0
        else 0.0
    )

    youden = sensitivity + specificity - 1.0

    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "precision": float(precision),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "f1": float(f1),
        "f2": float(f2),
        "youden": float(youden),
    }


def threshold_sweep(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, 1001),
                probabilities,
                np.array([0.5]),
            ]
        )
    )

    return pd.DataFrame(
        [
            metrics_at_threshold(
                y_true,
                probabilities,
                float(threshold),
            )
            for threshold in candidates
        ]
    )


def select_best_threshold(
    sweep: pd.DataFrame,
    metric: str,
) -> pd.Series:
    return (
        sweep.sort_values(
            by=[metric, "sensitivity", "fpr"],
            ascending=[False, False, True],
        )
        .iloc[0]
    )


def event_level_scores(
    feature_df: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    temp = feature_df[
        [
            "eventId",
            "userId",
            "typeStr",
            "label",
        ]
    ].copy()

    temp["probability"] = probabilities

    grouped = (
        temp.groupby("eventId", sort=False)
        .agg(
            userId=("userId", "first"),
            typeStr=("typeStr", "first"),
            label=("label", "first"),
            n_windows=("probability", "size"),
            probability_max=("probability", "max"),
            probability_mean=("probability", "mean"),
            probability_q70=(
                "probability",
                lambda values: float(
                    np.quantile(values, 0.70)
                ),
            ),
        )
        .reset_index()
    )

    return grouped


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [TRAIN_PATH, VAL_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró:\n{path}"
            )

    train_raw = pd.read_csv(
        TRAIN_PATH,
        low_memory=False,
    )

    val_raw = pd.read_csv(
        VAL_PATH,
        low_memory=False,
    )

    print("")
    print("Extrayendo características de entrenamiento...")
    train_features, feature_names = (
        create_feature_table(train_raw)
    )

    print("Extrayendo características de validación...")
    val_features, val_feature_names = (
        create_feature_table(val_raw)
    )

    if feature_names != val_feature_names:
        raise RuntimeError(
            "Las características de train y validation no coinciden."
        )

    x_train = train_features[
        feature_names
    ].to_numpy(dtype=np.float64)

    y_train = train_features[
        "label"
    ].to_numpy(dtype=np.int64)

    x_val = val_features[
        feature_names
    ].to_numpy(dtype=np.float64)

    y_val = val_features[
        "label"
    ].to_numpy(dtype=np.int64)

    sample_weight = event_balanced_weights(
        train_features
    )

    model_results: list[dict[str, Any]] = []
    trained_models: dict[float, Pipeline] = {}
    probability_by_c: dict[float, np.ndarray] = {}
    sweeps_by_c: dict[float, pd.DataFrame] = {}

    for c_value in C_VALUES:
        print(f"Entrenando C={c_value}...")

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        C=c_value,
                        penalty="l2",
                        solver="liblinear",
                        max_iter=5000,
                        random_state=42,
                    ),
                ),
            ]
        )

        pipeline.fit(
            x_train,
            y_train,
            logistic__sample_weight=sample_weight,
        )

        val_probabilities = pipeline.predict_proba(
            x_val
        )[:, 1]

        roc_auc = roc_auc_score(
            y_val,
            val_probabilities,
        )

        pr_auc = average_precision_score(
            y_val,
            val_probabilities,
        )

        sweep = threshold_sweep(
            y_val,
            val_probabilities,
        )

        best_youden = select_best_threshold(
            sweep,
            "youden",
        )

        best_f2 = select_best_threshold(
            sweep,
            "f2",
        )

        model_results.append(
            {
                "C": c_value,
                "roc_auc_window": roc_auc,
                "pr_auc_window": pr_auc,
                "best_youden": float(
                    best_youden["youden"]
                ),
                "youden_threshold": float(
                    best_youden["threshold"]
                ),
                "youden_sensitivity": float(
                    best_youden["sensitivity"]
                ),
                "youden_specificity": float(
                    best_youden["specificity"]
                ),
                "youden_fpr": float(
                    best_youden["fpr"]
                ),
                "best_f2": float(
                    best_f2["f2"]
                ),
                "f2_threshold": float(
                    best_f2["threshold"]
                ),
            }
        )

        trained_models[c_value] = pipeline
        probability_by_c[c_value] = (
            val_probabilities
        )
        sweeps_by_c[c_value] = sweep

    results_df = pd.DataFrame(
        model_results
    ).sort_values(
        by=[
            "roc_auc_window",
            "best_youden",
        ],
        ascending=[False, False],
    )

    best_c = float(
        results_df.iloc[0]["C"]
    )

    best_model = trained_models[best_c]
    best_probabilities = probability_by_c[best_c]
    best_sweep = sweeps_by_c[best_c]

    best_youden = select_best_threshold(
        best_sweep,
        "youden",
    )

    best_f2 = select_best_threshold(
        best_sweep,
        "f2",
    )

    # Evaluación por evento, sin usar test.
    event_scores = event_level_scores(
        val_features,
        best_probabilities,
    )

    event_y = event_scores[
        "label"
    ].to_numpy(dtype=np.int64)

    event_probability = event_scores[
        "probability_q70"
    ].to_numpy(dtype=np.float64)

    event_roc_auc = roc_auc_score(
        event_y,
        event_probability,
    )

    event_pr_auc = average_precision_score(
        event_y,
        event_probability,
    )

    event_sweep = threshold_sweep(
        event_y,
        event_probability,
    )

    event_best_youden = select_best_threshold(
        event_sweep,
        "youden",
    )

    # Coeficientes en escala estandarizada.
    logistic = best_model.named_steps["logistic"]
    coefficients = logistic.coef_[0]

    coefficient_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(
                coefficients
            ),
        }
    ).sort_values(
        by="absolute_coefficient",
        ascending=False,
    )

    # Guardado.
    model_path = OUTPUT_DIR / "baseline_fft_logreg.joblib"
    thresholds_path = OUTPUT_DIR / "umbrales_validacion.json"
    results_path = OUTPUT_DIR / "comparacion_C_validacion.csv"
    sweep_path = OUTPUT_DIR / "barrido_umbrales_ventana.csv"
    event_sweep_path = OUTPUT_DIR / "barrido_umbrales_evento.csv"
    coefficients_path = OUTPUT_DIR / "coeficientes_modelo.csv"
    train_features_path = OUTPUT_DIR / "features_train.csv"
    val_features_path = OUTPUT_DIR / "features_validation.csv"
    val_probabilities_path = OUTPUT_DIR / "probabilidades_validation.csv"
    event_scores_path = OUTPUT_DIR / "probabilidades_evento_validation.csv"
    summary_path = OUTPUT_DIR / "resumen_baseline_validacion.txt"

    joblib.dump(
        {
            "pipeline": best_model,
            "feature_names": feature_names,
            "fs_hz": FS_HZ,
            "n_samples": N_SAMPLES,
            "best_c": best_c,
        },
        model_path,
    )

    thresholds = {
        "best_c": best_c,
        "window_level": {
            "roc_auc": float(
                roc_auc_score(
                    y_val,
                    best_probabilities,
                )
            ),
            "pr_auc": float(
                average_precision_score(
                    y_val,
                    best_probabilities,
                )
            ),
            "youden": best_youden.to_dict(),
            "f2": best_f2.to_dict(),
        },
        "event_level_q70": {
            "roc_auc": float(event_roc_auc),
            "pr_auc": float(event_pr_auc),
            "youden": event_best_youden.to_dict(),
        },
    }

    thresholds_path.write_text(
        json.dumps(
            thresholds,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    best_sweep.to_csv(
        sweep_path,
        index=False,
    )

    event_sweep.to_csv(
        event_sweep_path,
        index=False,
    )

    coefficient_df.to_csv(
        coefficients_path,
        index=False,
    )

    train_features.to_csv(
        train_features_path,
        index=False,
    )

    val_features_with_prob = (
        val_features.copy()
    )

    val_features_with_prob[
        "probability_seizure"
    ] = best_probabilities

    val_features_with_prob.to_csv(
        val_probabilities_path,
        index=False,
    )

    event_scores.to_csv(
        event_scores_path,
        index=False,
    )

    # Gráficas.
    fpr_curve, tpr_curve, _ = roc_curve(
        y_val,
        best_probabilities,
    )

    plt.figure()
    plt.plot(
        fpr_curve,
        tpr_curve,
        label=(
            "ROC-AUC = "
            f"{roc_auc_score(y_val, best_probabilities):.3f}"
        ),
    )
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Sensibilidad")
    plt.title("Baseline FFT + regresión logística V2 — validación")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "curva_roc_ventanas.png",
        dpi=180,
    )
    plt.close()

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            y_val,
            best_probabilities,
        )
    )

    plt.figure()
    plt.plot(
        recall_curve,
        precision_curve,
        label=(
            "PR-AUC = "
            f"{average_precision_score(y_val, best_probabilities):.3f}"
        ),
    )
    plt.xlabel("Sensibilidad")
    plt.ylabel("Precisión")
    plt.title("Curva precisión-recall — validación")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "curva_precision_recall_ventanas.png",
        dpi=180,
    )
    plt.close()

    plt.figure()
    histogram_bins = np.linspace(0, 1, 21)
    plt.hist(
        best_probabilities[y_val == 0],
        bins=histogram_bins,
        alpha=0.6,
        label="No crisis",
    )
    plt.hist(
        best_probabilities[y_val == 1],
        bins=histogram_bins,
        alpha=0.6,
        label="Crisis",
    )
    plt.axvline(
        float(best_youden["threshold"]),
        linestyle="--",
        label="Umbral Youden",
    )
    plt.xlabel("Probabilidad de crisis")
    plt.ylabel("Ventanas")
    plt.title("Distribución de probabilidades — validación")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "histograma_probabilidades.png",
        dpi=180,
    )
    plt.close()

    summary_lines = [
        "BASELINE FFT + REGRESIÓN LOGÍSTICA — V2",
        "=" * 78,
        "",
        "Este análisis usa únicamente TRAIN y VALIDATION.",
        "El conjunto TEST permanece sin utilizar.",
        "",
        "DATOS",
        f"Train: {len(train_features)} ventanas, "
        f"{train_features['eventId'].nunique()} eventos",
        f"Validation: {len(val_features)} ventanas, "
        f"{val_features['eventId'].nunique()} eventos",
        f"Características: {len(feature_names)}",
        "",
        "SELECCIÓN DEL MODELO",
        f"Mejor C: {best_c}",
        f"ROC-AUC ventana: "
        f"{roc_auc_score(y_val, best_probabilities):.6f}",
        f"PR-AUC ventana: "
        f"{average_precision_score(y_val, best_probabilities):.6f}",
        "",
        "UMBRAL YOUDEN — VENTANAS",
    ]

    for key, value in best_youden.to_dict().items():
        summary_lines.append(f"{key}: {value}")

    summary_lines.extend(
        [
            "",
            "UMBRAL F2 — VENTANAS",
        ]
    )

    for key, value in best_f2.to_dict().items():
        summary_lines.append(f"{key}: {value}")

    summary_lines.extend(
        [
            "",
            "EVALUACIÓN POR EVENTO — CUANTIL 70 %",
            f"ROC-AUC evento: {event_roc_auc:.6f}",
            f"PR-AUC evento: {event_pr_auc:.6f}",
        ]
    )

    for key, value in event_best_youden.to_dict().items():
        summary_lines.append(f"{key}: {value}")

    summary_lines.extend(
        [
            "",
            "CARACTERÍSTICAS CON MAYOR PESO ABSOLUTO",
            coefficient_df.head(10).to_string(index=False),
            "",
            "COMPARACIÓN CON V1",
            "ROC-AUC ventana V1: 0.638444",
            "ROC-AUC evento V1: 0.578125",
            (
                "Cambio ROC-AUC ventana V2-V1: "
                f"{roc_auc_score(y_val, best_probabilities) - 0.638444:+.6f}"
            ),
            (
                "Cambio ROC-AUC evento V2-V1: "
                f"{event_roc_auc - 0.578125:+.6f}"
            ),
            "",
            "INTERPRETACIÓN",
        ]
    )

    window_auc = roc_auc_score(
        y_val,
        best_probabilities,
    )

    if window_auc < 0.55:
        summary_lines.append(
            "La separación sigue cercana al azar. "
            "Se deben revisar etiquetas/unidades antes de ampliar el modelo."
        )
    elif window_auc < 0.70:
        summary_lines.append(
            "Existe señal débil. Conviene ampliar datos y ajustar "
            "características antes de evaluar test."
        )
    else:
        summary_lines.append(
            "Existe una separación útil en validación. "
            "El siguiente paso sería fijar el umbral y evaluar test una sola vez."
        )

    summary_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("")
    print("=" * 78)
    print("BASELINE COMPLETADO")
    print("=" * 78)
    print(f"Mejor C: {best_c}")
    print(
        "ROC-AUC ventana: "
        f"{window_auc:.6f}"
    )
    print(
        "PR-AUC ventana: "
        f"{average_precision_score(y_val, best_probabilities):.6f}"
    )
    print(
        "ROC-AUC evento: "
        f"{event_roc_auc:.6f}"
    )
    print(
        "Umbral Youden ventana: "
        f"{float(best_youden['threshold']):.6f}"
    )
    print(f"Resumen: {summary_path}")
    print("")
    print("El conjunto TEST todavía no fue utilizado.")


if __name__ == "__main__":
    main()

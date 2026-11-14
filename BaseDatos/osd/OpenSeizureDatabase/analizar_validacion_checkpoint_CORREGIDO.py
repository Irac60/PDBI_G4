"""
analizar_validacion_checkpoint.py
=================================

Analiza el checkpoint ya entrenado SIN reentrenar y SIN usar testFeatures.csv.

Objetivos:
1. Cargar el checkpoint .pt.
2. Reconstruir la arquitectura DeepEpi PyTorch.
3. Obtener probabilidades sobre valFeatures.csv.
4. Calcular ROC-AUC y PR-AUC.
5. Barrer umbrales y buscar:
   - mejor Youden J;
   - mejor F2 (favorece sensibilidad).
6. Detectar si el modelo:
   - predice todo positivo;
   - predice todo negativo;
   - tiene ranking útil aunque el umbral 0.5 sea malo.
7. Guardar CSV, TXT y gráficas.

Ejecución:
    python .\analizar_validacion_checkpoint.py
"""

from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import importlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from user_tools.nnTraining2 import nnTrainer


# ============================================================
# RUTAS
# ============================================================

REPO_DIR = Path(r"C:\Users\cesar\osd\OpenSeizureDatabase")
PIPELINE_DIR = Path(r"C:\Users\cesar\osd\pipeline_prototipo")
RUN_DIR = PIPELINE_DIR / "entrenamiento_01"

CHECKPOINT_PATH = RUN_DIR / "prototipo_osdb_pytorch_smoke.pt"
CONFIG_PATH = RUN_DIR / "nnConfig_entrenamiento_01.json"
VALIDATION_PATH = RUN_DIR / "valFeatures.csv"

OUTPUT_DIR = RUN_DIR / "analisis_validacion"


# ============================================================
# UTILIDADES
# ============================================================

def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró:\n{path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def import_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_pred = (probabilities >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else float("nan")
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else float("nan")
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else float("nan")
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    accuracy = (tp + tn) / len(y_true)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    f1 = (
        2 * precision * sensitivity /
        (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )

    beta = 2.0
    f2 = (
        (1 + beta**2) * precision * sensitivity /
        (beta**2 * precision + sensitivity)
        if (beta**2 * precision + sensitivity) > 0
        else 0.0
    )

    youden = sensitivity + specificity - 1

    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "precision": float(precision),
        "f1": float(f1),
        "f2": float(f2),
        "youden": float(youden),
        "predicted_positive": int(y_pred.sum()),
        "predicted_negative": int((y_pred == 0).sum()),
    }


def choose_best(
    metrics_df: pd.DataFrame,
    column: str,
) -> pd.Series:
    # Desempate:
    # 1) mayor métrica;
    # 2) mayor sensibilidad;
    # 3) menor FPR.
    ordered = metrics_df.sort_values(
        by=[column, "sensitivity", "fpr"],
        ascending=[False, False, True],
    )

    return ordered.iloc[0]


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in [
        CHECKPOINT_PATH,
        CONFIG_PATH,
        VALIDATION_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"No se encontró:\n{path}")

    config = load_json(CONFIG_PATH)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model_class_path = config["modelConfig"]["modelClass"]
    wrapper_class = import_class(model_class_path)

    wrapper = wrapper_class(
        configObj=config["modelConfig"],
        debug=False,
    )

    model = wrapper.makeModel(
        input_shape=None,
        num_classes=2,
    )

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "El archivo .pt no contiene model_state_dict."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.cpu()

    validation_df = pd.read_csv(
        VALIDATION_PATH,
        low_memory=False,
    )

    x_list, y_list, used_indices = nnTrainer.df2trainingData(
        validation_df,
        wrapper,
        debug=False,
        return_row_indices=True,
    )

    x = np.asarray(x_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    if x.ndim != 2 or x.shape[1] != 750:
        raise RuntimeError(
            f"Forma inesperada de entrada: {x.shape}; se esperaba (N, 750)."
        )

    probabilities_parts: list[np.ndarray] = []

    batch_size = 128

    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))

            batch = torch.from_numpy(x[start:stop]).float()

            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1]

            probabilities_parts.append(
                probs.cpu().numpy()
            )

    probabilities = np.concatenate(probabilities_parts).astype(
        np.float64,
        copy=False,
    )

    # Softmax debería producir valores finitos en [0, 1].
    # Esta comprobación evita que una salida no finita llegue a las métricas.
    if not np.isfinite(probabilities).all():
        invalid_count = int((~np.isfinite(probabilities)).sum())
        raise RuntimeError(
            f"Se encontraron {invalid_count} probabilidades no finitas."
        )

    probabilities = np.clip(probabilities, 0.0, 1.0)

    if len(probabilities) != len(y):
        raise RuntimeError("Probabilidades y etiquetas tienen distinta longitud.")

    roc_auc = roc_auc_score(y, probabilities)
    pr_auc = average_precision_score(y, probabilities)

    # Umbrales candidatos:
    # - malla regular;
    # - todos los valores únicos de probabilidad;
    # - 0.5 explícito.
    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, 1001),
                probabilities,
                np.array([0.5]),
            ]
        )
    )

    rows = [
        binary_metrics(y, probabilities, float(t))
        for t in thresholds
    ]

    metrics_df = pd.DataFrame(rows)

    at_half = metrics_df.iloc[
        (metrics_df["threshold"] - 0.5).abs().argmin()
    ]

    best_youden = choose_best(metrics_df, "youden")
    best_f2 = choose_best(metrics_df, "f2")

    probability_df = validation_df.iloc[used_indices].copy()
    probability_df.insert(0, "source_row_index", used_indices)
    probability_df["true_label"] = y
    probability_df["probability_seizure"] = probabilities
    probability_df["pred_at_0_5"] = (
        probabilities >= 0.5
    ).astype(int)
    probability_df["pred_at_best_youden"] = (
        probabilities >= float(best_youden["threshold"])
    ).astype(int)
    probability_df["pred_at_best_f2"] = (
        probabilities >= float(best_f2["threshold"])
    ).astype(int)

    probability_path = OUTPUT_DIR / "probabilidades_validacion.csv"
    threshold_path = OUTPUT_DIR / "barrido_umbrales_validacion.csv"
    summary_path = OUTPUT_DIR / "resumen_validacion_checkpoint.txt"
    roc_path = OUTPUT_DIR / "curva_roc_validacion.png"
    pr_path = OUTPUT_DIR / "curva_precision_recall_validacion.png"
    histogram_path = OUTPUT_DIR / "histograma_probabilidades_validacion.png"
    chosen_path = OUTPUT_DIR / "umbrales_seleccionados.json"

    probability_df.to_csv(probability_path, index=False)
    metrics_df.to_csv(threshold_path, index=False)

    chosen = {
        "checkpoint": str(CHECKPOINT_PATH),
        "roc_auc_validation": float(roc_auc),
        "pr_auc_validation": float(pr_auc),
        "threshold_default_0_5": at_half.to_dict(),
        "best_youden": best_youden.to_dict(),
        "best_f2": best_f2.to_dict(),
    }

    chosen_path.write_text(
        json.dumps(chosen, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Gráfica ROC
    fpr_curve, tpr_curve, _ = roc_curve(y, probabilities)

    plt.figure()
    plt.plot(fpr_curve, tpr_curve, label=f"ROC-AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", label="Azar")
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Sensibilidad")
    plt.title("Curva ROC — validación")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_path, dpi=180)
    plt.close()

    # Gráfica precision-recall
    precision_curve, recall_curve, _ = precision_recall_curve(
        y,
        probabilities,
    )

    plt.figure()
    plt.plot(
        recall_curve,
        precision_curve,
        label=f"PR-AUC = {pr_auc:.3f}",
    )
    plt.xlabel("Sensibilidad / recall")
    plt.ylabel("Precisión")
    plt.title("Curva precisión-recall — validación")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pr_path, dpi=180)
    plt.close()

    # Histograma por clase.
    # Se usan bordes fijos en [0, 1] porque, tras solo tres épocas,
    # una clase puede recibir probabilidades idénticas o casi idénticas.
    # np.histogram con bins=20 intenta inferir el rango y puede fallar
    # cuando ese rango es extremadamente pequeño.
    histogram_bins = np.linspace(0.0, 1.0, 21)

    probabilities_negative = probabilities[y == 0]
    probabilities_positive = probabilities[y == 1]

    plt.figure()
    plt.hist(
        probabilities_negative,
        bins=histogram_bins,
        alpha=0.6,
        label="No crisis",
    )
    plt.hist(
        probabilities_positive,
        bins=histogram_bins,
        alpha=0.6,
        label="Crisis",
    )
    plt.axvline(0.5, linestyle="--", label="Umbral 0.5")
    plt.axvline(
        float(best_youden["threshold"]),
        linestyle=":",
        label="Mejor Youden",
    )
    plt.xlabel("Probabilidad de crisis")
    plt.ylabel("Número de ventanas")
    plt.title("Distribución de probabilidades — validación")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(histogram_path, dpi=180)
    plt.close()

    lines = [
        "ANÁLISIS DEL CHECKPOINT SOBRE VALIDACIÓN",
        "=" * 78,
        "",
        f"Checkpoint: {CHECKPOINT_PATH}",
        f"Vectores evaluados: {len(y)}",
        f"Positivos reales: {int((y == 1).sum())}",
        f"Negativos reales: {int((y == 0).sum())}",
        f"ROC-AUC: {roc_auc:.6f}",
        f"PR-AUC: {pr_auc:.6f}",
        f"Probabilidad mínima global: {probabilities.min():.12f}",
        f"Probabilidad máxima global: {probabilities.max():.12f}",
        f"Probabilidad media global: {probabilities.mean():.12f}",
        f"Rango global: {(probabilities.max() - probabilities.min()):.12e}",
        "",
        "PROBABILIDADES POR CLASE",
        (
            "No crisis: "
            f"min={probabilities_negative.min():.12f}, "
            f"max={probabilities_negative.max():.12f}, "
            f"media={probabilities_negative.mean():.12f}, "
            f"DE={probabilities_negative.std():.12e}"
        ),
        (
            "Crisis: "
            f"min={probabilities_positive.min():.12f}, "
            f"max={probabilities_positive.max():.12f}, "
            f"media={probabilities_positive.mean():.12f}, "
            f"DE={probabilities_positive.std():.12e}"
        ),
        "",
        "UMBRAL 0.5",
    ]

    for key, value in at_half.to_dict().items():
        lines.append(f"{key}: {value}")

    lines.extend(["", "MEJOR UMBRAL SEGÚN YOUDEN"])

    for key, value in best_youden.to_dict().items():
        lines.append(f"{key}: {value}")

    lines.extend(["", "MEJOR UMBRAL SEGÚN F2"])

    for key, value in best_f2.to_dict().items():
        lines.append(f"{key}: {value}")

    lines.extend(
        [
            "",
            "INTERPRETACIÓN AUTOMÁTICA",
        ]
    )

    if roc_auc < 0.55:
        lines.append(
            "El ranking es cercano al azar; ajustar el umbral no basta."
        )
        lines.append(
            "Conviene aumentar datos y reentrenar antes de usar test."
        )
    elif roc_auc < 0.70:
        lines.append(
            "Existe señal débil; el umbral puede mejorar el resultado,"
        )
        lines.append(
            "pero el modelo todavía requiere reentrenamiento."
        )
    else:
        lines.append(
            "Existe separación útil en validación; puede fijarse un umbral"
        )
        lines.append(
            "y luego evaluarse una sola vez sobre test."
        )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("")
    print("=" * 78)
    print("ANÁLISIS DE VALIDACIÓN COMPLETADO")
    print("=" * 78)
    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"PR-AUC: {pr_auc:.6f}")
    print(
        "Umbral Youden: "
        f"{float(best_youden['threshold']):.6f}"
    )
    print(
        "Sensibilidad Youden: "
        f"{float(best_youden['sensitivity']):.6f}"
    )
    print(
        "FPR Youden: "
        f"{float(best_youden['fpr']):.6f}"
    )
    print(f"Resumen: {summary_path}")
    print(f"Probabilidades: {probability_path}")
    print("")
    print("El conjunto TEST todavía no fue utilizado.")


if __name__ == "__main__":
    main()

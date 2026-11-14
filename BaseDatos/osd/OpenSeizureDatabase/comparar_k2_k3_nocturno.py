"""
comparar_k2_k3_nocturno.py
==========================

Compara k=2 y k=3 usando LOS MISMOS umbrales deterministas ya seleccionados.

Importante:
- No reoptimiza power_threshold.
- No reoptimiza ratio_threshold.
- Usa TRAIN y VALIDATION.
- TEST permanece cerrado.
- El objetivo es decidir si k=2 puede actuar como WARNING y k=3 como ALARM.

Umbrales fijos:
power_threshold = 0.05425036036486526
ratio_threshold = 0.436484300104836
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PIPELINE_DIR = Path(
    r"C:\Users\cesar\osd\dataset_nocturno_osdb_v1"
    r"\pipeline_nocturno_v2_limpio"
)

FEATURE_DIR = (
    PIPELINE_DIR
    / "baseline_fft_logreg_nocturno_v3_balance_usuario"
)

TRAIN_PATH = FEATURE_DIR / "features_train.csv"
VAL_PATH = FEATURE_DIR / "probabilidades_validation.csv"

OUTPUT_DIR = PIPELINE_DIR / "comparacion_k2_k3_nocturno"

POWER_THRESHOLD = 0.05425036036486526
RATIO_THRESHOLD = 0.436484300104836


def normalize_event_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        head = text[:-2]
        if head.replace("-", "").isdigit():
            text = head
    return text


def normalize_label(value: Any) -> int:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "1.0", "seizure"}:
            return 1
        if text in {"0", "0.0", "false alarm", "nda", "non-seizure", "nonseizure"}:
            return 0

    number = float(value)
    if number == 1:
        return 1
    if number == 0:
        return 0

    raise ValueError(f"Etiqueta no binaria: {value!r}")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "eventId",
        "label",
        "power_3_8",
        "ratio_3_8",
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            "Faltan columnas: " + ", ".join(sorted(missing))
        )

    out = df.copy()

    out["event_id_norm"] = out["eventId"].map(
        normalize_event_id
    )

    out["label_norm"] = out["label"].map(
        normalize_label
    )

    if "source_row" in out.columns:
        out = out.sort_values(
            by=["event_id_norm", "source_row"]
        )
    else:
        out["_order"] = (
            out.groupby("event_id_norm")
            .cumcount()
        )
        out = out.sort_values(
            by=["event_id_norm", "_order"]
        )

    return out


def longest_true_run(values: np.ndarray) -> int:
    best = 0
    current = 0

    for value in values.astype(bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return best


def evaluate(
    df: pd.DataFrame,
    k: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []

    for event_id, group in df.groupby(
        "event_id_norm",
        sort=False,
    ):
        positive = (
            (group["power_3_8"].to_numpy() >= POWER_THRESHOLD)
            &
            (group["ratio_3_8"].to_numpy() >= RATIO_THRESHOLD)
        )

        longest = longest_true_run(positive)
        prediction = int(longest >= k)
        true_label = int(group["label_norm"].iloc[0])

        alarm_window = np.nan

        if prediction:
            run = 0
            for idx, flag in enumerate(positive, start=1):
                if flag:
                    run += 1
                else:
                    run = 0

                if run >= k:
                    alarm_window = idx
                    break

        rows.append(
            {
                "eventId": event_id,
                "true_label": true_label,
                "predicted_label": prediction,
                "n_windows": len(group),
                "positive_windows": int(positive.sum()),
                "longest_positive_run": longest,
                "alarm_time_from_first_window_s": (
                    alarm_window * 5
                    if not np.isnan(alarm_window)
                    else np.nan
                ),
            }
        )

    details = pd.DataFrame(rows)

    y = details["true_label"].to_numpy()
    p = details["predicted_label"].to_numpy()

    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = 1.0 - specificity
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    balanced_accuracy = (sensitivity + specificity) / 2.0

    detected_times = details.loc[
        (details["true_label"] == 1)
        & (details["predicted_label"] == 1),
        "alarm_time_from_first_window_s",
    ].dropna()

    metrics = {
        "k": k,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "fpr": fpr,
        "precision": precision,
        "balanced_accuracy": balanced_accuracy,
        "median_alarm_time_s": (
            float(detected_times.median())
            if not detected_times.empty
            else np.nan
        ),
    }

    return metrics, details


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

    train = prepare(
        pd.read_csv(TRAIN_PATH, low_memory=False)
    )

    validation = prepare(
        pd.read_csv(VAL_PATH, low_memory=False)
    )

    summary_rows = []

    detail_outputs = {}

    for split_name, df in [
        ("train", train),
        ("validation", validation),
    ]:
        for k in [2, 3]:
            metrics, details = evaluate(
                df,
                k=k,
            )

            row = {
                "split": split_name,
                **metrics,
            }

            summary_rows.append(row)
            detail_outputs[
                (split_name, k)
            ] = details

    comparison = pd.DataFrame(summary_rows)

    comparison_path = (
        OUTPUT_DIR / "comparacion_k2_k3.csv"
    )

    summary_path = (
        OUTPUT_DIR / "resumen_comparacion_k2_k3.txt"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    for (split_name, k), details in detail_outputs.items():
        details.to_csv(
            OUTPUT_DIR
            / f"detalle_{split_name}_k{k}.csv",
            index=False,
        )

    val_k2 = comparison.loc[
        (comparison["split"] == "validation")
        & (comparison["k"] == 2)
    ].iloc[0]

    val_k3 = comparison.loc[
        (comparison["split"] == "validation")
        & (comparison["k"] == 3)
    ].iloc[0]

    lines = [
        "COMPARACIÓN TEMPORAL k=2 vs k=3 — NOCTURNO",
        "=" * 78,
        "",
        "Umbrales fijos:",
        f"power_threshold = {POWER_THRESHOLD}",
        f"ratio_threshold = {RATIO_THRESHOLD}",
        "",
        "No se reoptimizan umbrales.",
        "TEST permanece cerrado.",
        "",
        "RESULTADOS",
        comparison.to_string(index=False),
        "",
        "DECISIÓN ORIENTATIVA",
    ]

    if (
        val_k2["sensitivity"] >= val_k3["sensitivity"]
        and val_k2["fpr"] <= val_k3["fpr"]
    ):
        lines.extend(
            [
                "k=2 domina o iguala a k=3 en VALIDATION.",
                (
                    "Para este prototipo puede usarse k=2 como WARNING "
                    "de movimiento sospechoso."
                ),
                (
                    "La confirmación final deberá venir después de PPG/EEG "
                    "o de una condición adicional."
                ),
            ]
        )
    else:
        lines.extend(
            [
                (
                    "k=2 mejora sensibilidad pero aumenta falsas alarmas "
                    "respecto de k=3."
                ),
                (
                    "Conviene mantener k=2 como WARNING y k=3 como ALARM."
                ),
            ]
        )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("")
    print("=" * 78)
    print("COMPARACIÓN k=2 vs k=3 COMPLETADA")
    print("=" * 78)
    print(comparison.to_string(index=False))
    print("")
    print(f"Resumen: {summary_path}")
    print("")
    print("TEST permanece cerrado.")


if __name__ == "__main__":
    main()

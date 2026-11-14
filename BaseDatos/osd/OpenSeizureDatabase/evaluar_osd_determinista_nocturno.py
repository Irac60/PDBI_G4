"""
evaluar_osd_determinista_nocturno.py
====================================

Evalúa un detector determinista tipo OSD sobre el dataset nocturno V2 limpio.

Usa SOLO:
- TRAIN para seleccionar umbrales.
- VALIDATION para evaluar la regla seleccionada.
- TEST permanece cerrado.

Regla por ventana:
    seizure_like =
        power_3_8 >= power_threshold
        AND
        ratio_3_8 >= ratio_threshold

Regla temporal:
    k = 2 o 3 ventanas consecutivas positivas.

Selección:
1. Se prueban combinaciones de umbrales usando TRAIN.
2. Se prioriza sensibilidad por evento >= 0.90.
3. Entre las que cumplen, se elige la menor FPR.
4. Si ninguna cumple, se usa mejor Youden.
5. Luego esa regla FIJA se evalúa sobre VALIDATION.

Entradas:
C:\\Users\\cesar\\osd\\dataset_nocturno_osdb_v1\\pipeline_nocturno_v2_limpio\\
    baseline_fft_logreg_nocturno_v3_balance_usuario\\features_train.csv
    baseline_fft_logreg_nocturno_v3_balance_usuario\\features_validation.csv

Salida:
...\\osd_determinista_nocturno\\

NO usa TEST.
"""

from __future__ import annotations

import json
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

TRAIN_FEATURES = FEATURE_DIR / "features_train.csv"
VAL_FEATURES = FEATURE_DIR / "features_validation.csv"

OUTPUT_DIR = PIPELINE_DIR / "osd_determinista_nocturno"

TARGET_SENSITIVITY = 0.90

# Regla temporal OSD clásica: 2 = warning, 3 = alarm.
K_VALUES = [2, 3]


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
            "Faltan columnas: "
            + ", ".join(sorted(missing))
        )

    out = df.copy()

    out["event_id_norm"] = (
        out["eventId"]
        .map(normalize_event_id)
    )

    out["label_norm"] = (
        out["label"]
        .map(normalize_label)
    )

    # Mantener orden temporal.
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


def evaluate_event_rule(
    df: pd.DataFrame,
    power_threshold: float,
    ratio_threshold: float,
    k: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    details = []

    for event_id, group in df.groupby(
        "event_id_norm",
        sort=False,
    ):
        window_positive = (
            (group["power_3_8"].to_numpy() >= power_threshold)
            &
            (group["ratio_3_8"].to_numpy() >= ratio_threshold)
        )

        longest_run = longest_true_run(
            window_positive
        )

        prediction = int(
            longest_run >= k
        )

        true_label = int(
            group["label_norm"].iloc[0]
        )

        first_alarm_window = None

        if prediction == 1:
            run = 0

            for index, flag in enumerate(
                window_positive,
                start=1,
            ):
                if flag:
                    run += 1
                else:
                    run = 0

                if run >= k:
                    first_alarm_window = index
                    break

        details.append(
            {
                "eventId": event_id,
                "true_label": true_label,
                "predicted_label": prediction,
                "n_windows": len(group),
                "positive_windows": int(
                    window_positive.sum()
                ),
                "longest_positive_run": longest_run,
                "alarm_time_from_first_window_s": (
                    first_alarm_window * 5
                    if first_alarm_window is not None
                    else np.nan
                ),
                "mean_power_3_8": float(
                    group["power_3_8"].mean()
                ),
                "max_power_3_8": float(
                    group["power_3_8"].max()
                ),
                "mean_ratio_3_8": float(
                    group["ratio_3_8"].mean()
                ),
                "max_ratio_3_8": float(
                    group["ratio_3_8"].max()
                ),
            }
        )

    detail_df = pd.DataFrame(details)

    y = detail_df["true_label"].to_numpy()
    pred = detail_df["predicted_label"].to_numpy()

    tp = int(
        ((y == 1) & (pred == 1)).sum()
    )
    tn = int(
        ((y == 0) & (pred == 0)).sum()
    )
    fp = int(
        ((y == 0) & (pred == 1)).sum()
    )
    fn = int(
        ((y == 1) & (pred == 0)).sum()
    )

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

    fpr = 1.0 - specificity

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    balanced_accuracy = (
        (sensitivity + specificity) / 2.0
    )

    youden = (
        sensitivity + specificity - 1.0
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
        / (
            beta**2 * precision
            + sensitivity
        )
        if (
            beta**2 * precision
            + sensitivity
        ) > 0
        else 0.0
    )

    detected_positive_times = (
        detail_df.loc[
            (detail_df["true_label"] == 1)
            & (detail_df["predicted_label"] == 1),
            "alarm_time_from_first_window_s",
        ]
        .dropna()
        .to_numpy()
    )

    metrics = {
        "power_threshold": float(
            power_threshold
        ),
        "ratio_threshold": float(
            ratio_threshold
        ),
        "required_consecutive": int(k),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "sensitivity": float(
            sensitivity
        ),
        "specificity": float(
            specificity
        ),
        "fpr": float(fpr),
        "precision": float(
            precision
        ),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "f1": float(f1),
        "f2": float(f2),
        "youden": float(youden),
        "median_alarm_time_s": (
            float(
                np.median(
                    detected_positive_times
                )
            )
            if detected_positive_times.size
            else np.nan
        ),
    }

    return metrics, detail_df


def candidate_thresholds(
    values: pd.Series,
) -> np.ndarray:
    """
    Usa cuantiles de TRAIN para evitar una cuadrícula enorme.
    """
    clean = (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .dropna()
        .to_numpy()
    )

    quantiles = np.linspace(
        0.05,
        0.95,
        19,
    )

    thresholds = np.unique(
        np.quantile(
            clean,
            quantiles,
        )
    )

    return thresholds


def select_rule(
    sweep: pd.DataFrame,
) -> pd.Series:
    eligible = sweep.loc[
        sweep["sensitivity"]
        >= TARGET_SENSITIVITY
    ].copy()

    if not eligible.empty:
        return (
            eligible.sort_values(
                by=[
                    "fpr",
                    "balanced_accuracy",
                    "youden",
                    "required_consecutive",
                ],
                ascending=[
                    True,
                    False,
                    False,
                    True,
                ],
            )
            .iloc[0]
        )

    return (
        sweep.sort_values(
            by=[
                "youden",
                "sensitivity",
                "fpr",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .iloc[0]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        TRAIN_FEATURES,
        VAL_FEATURES,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró:\n{path}"
            )

    train = prepare(
        pd.read_csv(
            TRAIN_FEATURES,
            low_memory=False,
        )
    )

    validation = prepare(
        pd.read_csv(
            VAL_FEATURES,
            low_memory=False,
        )
    )

    power_thresholds = (
        candidate_thresholds(
            train["power_3_8"]
        )
    )

    ratio_thresholds = (
        candidate_thresholds(
            train["ratio_3_8"]
        )
    )

    rows = []

    print("")
    print(
        "Buscando regla determinista "
        "solo sobre TRAIN..."
    )

    for k in K_VALUES:
        for power_threshold in power_thresholds:
            for ratio_threshold in ratio_thresholds:
                metrics, _ = evaluate_event_rule(
                    train,
                    power_threshold=float(
                        power_threshold
                    ),
                    ratio_threshold=float(
                        ratio_threshold
                    ),
                    k=k,
                )

                rows.append(metrics)

    sweep = pd.DataFrame(rows)

    selected = select_rule(
        sweep
    )

    selected_power = float(
        selected["power_threshold"]
    )

    selected_ratio = float(
        selected["ratio_threshold"]
    )

    selected_k = int(
        selected["required_consecutive"]
    )

    train_metrics, train_detail = (
        evaluate_event_rule(
            train,
            selected_power,
            selected_ratio,
            selected_k,
        )
    )

    val_metrics, val_detail = (
        evaluate_event_rule(
            validation,
            selected_power,
            selected_ratio,
            selected_k,
        )
    )

    sweep_path = (
        OUTPUT_DIR
        / "barrido_umbral_osd_train.csv"
    )

    rule_path = (
        OUTPUT_DIR
        / "regla_osd_seleccionada.json"
    )

    train_detail_path = (
        OUTPUT_DIR
        / "detalle_eventos_train.csv"
    )

    val_detail_path = (
        OUTPUT_DIR
        / "detalle_eventos_validation.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "resumen_osd_determinista_nocturno.txt"
    )

    sweep.to_csv(
        sweep_path,
        index=False,
    )

    train_detail.to_csv(
        train_detail_path,
        index=False,
    )

    val_detail.to_csv(
        val_detail_path,
        index=False,
    )

    rule_payload = {
        "selected_using": "TRAIN only",
        "target_sensitivity": (
            TARGET_SENSITIVITY
        ),
        "power_threshold": (
            selected_power
        ),
        "ratio_threshold": (
            selected_ratio
        ),
        "required_consecutive": (
            selected_k
        ),
        "train_metrics": (
            train_metrics
        ),
        "validation_metrics": (
            val_metrics
        ),
    }

    rule_path.write_text(
        json.dumps(
            rule_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "OSD DETERMINISTA — DATASET NOCTURNO",
        "=" * 78,
        "",
        (
            "TRAIN se utilizó para seleccionar "
            "los umbrales."
        ),
        (
            "VALIDATION se utilizó únicamente "
            "para evaluar la regla ya fijada."
        ),
        "TEST permanece cerrado.",
        "",
        "REGLA POR VENTANA",
        (
            "seizure_like = "
            "power_3_8 >= power_threshold "
            "AND ratio_3_8 >= ratio_threshold"
        ),
        "",
        "REGLA SELECCIONADA",
        f"power_threshold: {selected_power}",
        f"ratio_threshold: {selected_ratio}",
        f"required_consecutive: {selected_k}",
        "",
        "RESULTADO TRAIN",
    ]

    for key, value in train_metrics.items():
        lines.append(
            f"{key}: {value}"
        )

    lines.extend(
        [
            "",
            "RESULTADO VALIDATION",
        ]
    )

    for key, value in val_metrics.items():
        lines.append(
            f"{key}: {value}"
        )

    lines.extend(
        [
            "",
            "DETALLE VALIDATION",
            val_detail.to_string(
                index=False
            ),
            "",
            "INTERPRETACIÓN",
        ]
    )

    if (
        val_metrics["sensitivity"] >= 0.80
        and val_metrics["fpr"] <= 0.50
    ):
        lines.append(
            "La regla determinista conserva "
            "sensibilidad útil como detector "
            "de movimiento nocturno sospechoso."
        )
    else:
        lines.append(
            "La acelerometría nocturna por sí "
            "sola sigue siendo insuficiente "
            "como decisión final; debe usarse "
            "como trigger para PPG/EEG."
        )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("")
    print("=" * 78)
    print("OSD DETERMINISTA COMPLETADO")
    print("=" * 78)
    print(
        f"power_threshold = "
        f"{selected_power}"
    )
    print(
        f"ratio_threshold = "
        f"{selected_ratio}"
    )
    print(
        f"k = {selected_k}"
    )
    print("")
    print(
        "VALIDATION:"
    )
    print(
        f"Sensibilidad = "
        f"{val_metrics['sensitivity']:.3f}"
    )
    print(
        f"Especificidad = "
        f"{val_metrics['specificity']:.3f}"
    )
    print(
        f"FPR = "
        f"{val_metrics['fpr']:.3f}"
    )
    print(
        f"TP={val_metrics['tp']}, "
        f"FP={val_metrics['fp']}, "
        f"FN={val_metrics['fn']}, "
        f"TN={val_metrics['tn']}"
    )
    print("")
    print(f"Resumen: {summary_path}")
    print("")
    print("TEST permanece cerrado.")


if __name__ == "__main__":
    main()

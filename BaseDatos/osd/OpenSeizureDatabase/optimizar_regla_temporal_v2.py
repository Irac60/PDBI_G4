
"""
optimizar_regla_temporal_v2.py
==============================

Optimiza una regla temporal sobre las probabilidades del baseline FFT V2,
utilizando SOLO el conjunto de VALIDACIÓN.

Regla:
- Una ventana de 5 s es positiva si p(crisis) >= umbral.
- WARNING candidato: 2 ventanas positivas consecutivas.
- ALARM candidato: k ventanas positivas consecutivas.
- Se barren k = 1, 2, 3 y 4, y múltiples umbrales.

Objetivos:
1. Medir sensibilidad y FPR por EVENTO.
2. Buscar la regla con menor FPR manteniendo sensibilidad >= 0.80.
3. Reportar también la mejor regla por índice de Youden.
4. Calcular tiempo aproximado hasta la alarma desde la primera ventana
   conservada del evento.
5. NO abrir ni utilizar testData.csv.

Entrada:
C:\\Users\\cesar\\osd\\pipeline_prototipo_v2\\baseline_fft_logreg_v2\\
probabilidades_validation.csv

Salidas:
- barrido_reglas_temporales.csv
- detalle_eventos_mejor_regla.csv
- reglas_candidatas_validacion.json
- resumen_regla_temporal_validacion.txt
- curva_sensibilidad_fpr_temporal.png

Ejecución:
    python .\\optimizar_regla_temporal_v2.py
"""

from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# ============================================================
# RUTAS Y CONFIGURACIÓN
# ============================================================

BASELINE_DIR = Path(
    r"C:\Users\cesar\osd\pipeline_prototipo_v2"
    r"\baseline_fft_logreg_v2"
)

INPUT_PATH = BASELINE_DIR / "probabilidades_validation.csv"
OUTPUT_DIR = BASELINE_DIR / "regla_temporal_validacion"

WINDOW_SECONDS = 5.0
MIN_SENSITIVITY_TARGET = 0.80
CONSECUTIVE_OPTIONS = [1, 2, 3, 4]


# ============================================================
# FUNCIONES
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


def first_consecutive_run_end(
    flags: np.ndarray,
    required_consecutive: int,
) -> int | None:
    """
    Devuelve el índice 0-based de la ventana donde termina la primera
    secuencia de required_consecutive positivos.
    """
    run_length = 0

    for index, flag in enumerate(flags):
        if bool(flag):
            run_length += 1

            if run_length >= required_consecutive:
                return index
        else:
            run_length = 0

    return None


def longest_positive_run(flags: np.ndarray) -> int:
    best = 0
    current = 0

    for flag in flags:
        if bool(flag):
            current += 1
            best = max(best, current)
        else:
            current = 0

    return best


def event_predictions(
    df: pd.DataFrame,
    threshold: float,
    required_consecutive: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for event_id, event_df in df.groupby(
        "eventId",
        sort=False,
        dropna=False,
    ):
        event_df = event_df.sort_values(
            by="source_row",
            kind="stable",
        )

        labels = event_df["label"].to_numpy(
            dtype=np.int64,
        )

        if np.unique(labels).size != 1:
            raise RuntimeError(
                f"El evento {event_id} contiene etiquetas mezcladas."
            )

        probabilities = event_df[
            "probability_seizure"
        ].to_numpy(dtype=np.float64)

        flags = probabilities >= threshold

        end_index = first_consecutive_run_end(
            flags,
            required_consecutive,
        )

        predicted = int(end_index is not None)

        # El detector decide al terminar la ventana indicada.
        # Ej.: k=3 y run desde la primera ventana -> 15 s.
        alarm_time_s = (
            float((end_index + 1) * WINDOW_SECONDS)
            if end_index is not None
            else np.nan
        )

        rows.append(
            {
                "eventId": event_id,
                "userId": event_df["userId"].iloc[0],
                "typeStr": event_df["typeStr"].iloc[0],
                "true_label": int(labels[0]),
                "predicted_label": predicted,
                "n_windows": len(event_df),
                "probability_min": float(
                    probabilities.min()
                ),
                "probability_mean": float(
                    probabilities.mean()
                ),
                "probability_max": float(
                    probabilities.max()
                ),
                "positive_window_count": int(
                    flags.sum()
                ),
                "positive_window_fraction": float(
                    flags.mean()
                ),
                "longest_positive_run": int(
                    longest_positive_run(flags)
                ),
                "first_alarm_end_window_1based": (
                    int(end_index + 1)
                    if end_index is not None
                    else np.nan
                ),
                "alarm_time_from_first_retained_window_s": (
                    alarm_time_s
                ),
            }
        )

    return pd.DataFrame(rows)


def event_metrics(
    event_df: pd.DataFrame,
    threshold: float,
    required_consecutive: int,
) -> dict[str, Any]:
    y_true = event_df["true_label"].to_numpy(
        dtype=np.int64,
    )

    y_pred = event_df["predicted_label"].to_numpy(
        dtype=np.int64,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    fpr = 1.0 - specificity

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    balanced_accuracy = (
        sensitivity + specificity
    ) / 2.0

    f1 = (
        2.0 * precision * sensitivity
        / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )

    beta = 2.0

    f2 = (
        (1.0 + beta**2)
        * precision
        * sensitivity
        / (beta**2 * precision + sensitivity)
        if (
            beta**2 * precision + sensitivity
        ) > 0
        else 0.0
    )

    youden = sensitivity + specificity - 1.0

    detected_positive = event_df[
        (event_df["true_label"] == 1)
        & (event_df["predicted_label"] == 1)
    ]

    alarm_times = detected_positive[
        "alarm_time_from_first_retained_window_s"
    ].dropna()

    return {
        "threshold": float(threshold),
        "required_consecutive": int(
            required_consecutive
        ),
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
        "predicted_positive_events": int(
            y_pred.sum()
        ),
        "predicted_negative_events": int(
            (y_pred == 0).sum()
        ),
        "median_alarm_time_s_detected_positives": (
            float(alarm_times.median())
            if not alarm_times.empty
            else np.nan
        ),
        "mean_alarm_time_s_detected_positives": (
            float(alarm_times.mean())
            if not alarm_times.empty
            else np.nan
        ),
    }


def choose_best_youden(
    sweep: pd.DataFrame,
) -> pd.Series:
    return (
        sweep.sort_values(
            by=[
                "youden",
                "sensitivity",
                "fpr",
                "median_alarm_time_s_detected_positives",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
        )
        .iloc[0]
    )


def choose_target_sensitivity(
    sweep: pd.DataFrame,
    minimum_sensitivity: float,
) -> pd.Series | None:
    candidates = sweep[
        sweep["sensitivity"]
        >= minimum_sensitivity
    ].copy()

    if candidates.empty:
        return None

    # Prioridad:
    # 1) menor FPR;
    # 2) mayor sensibilidad;
    # 3) menor tiempo de alarma;
    # 4) menor número de ventanas consecutivas;
    # 5) mayor umbral.
    return (
        candidates.sort_values(
            by=[
                "fpr",
                "sensitivity",
                "median_alarm_time_s_detected_positives",
                "required_consecutive",
                "threshold",
            ],
            ascending=[
                True,
                False,
                True,
                True,
                False,
            ],
        )
        .iloc[0]
    )


def row_to_json_dict(
    row: pd.Series | None,
) -> dict[str, Any] | None:
    if row is None:
        return None

    result: dict[str, Any] = {}

    for key, value in row.to_dict().items():
        if pd.isna(value):
            result[key] = None
        elif isinstance(value, (np.integer,)):
            result[key] = int(value)
        elif isinstance(value, (np.floating,)):
            result[key] = float(value)
        else:
            result[key] = value

    return result


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró:\n{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    required_columns = {
        "source_row",
        "eventId",
        "userId",
        "typeStr",
        "label",
        "probability_seizure",
    }

    missing = required_columns.difference(
        df.columns
    )

    if missing:
        raise RuntimeError(
            "Faltan columnas requeridas:\n"
            + "\n".join(sorted(missing))
        )

    df = df.copy()

    df["label"] = df["label"].map(
        normalize_label
    )

    df["probability_seizure"] = pd.to_numeric(
        df["probability_seizure"],
        errors="raise",
    )

    if not np.isfinite(
        df["probability_seizure"]
    ).all():
        raise RuntimeError(
            "Existen probabilidades no finitas."
        )

    probabilities = df[
        "probability_seizure"
    ].to_numpy(dtype=np.float64)

    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, 1001),
                probabilities,
            ]
        )
    )

    rows: list[dict[str, Any]] = []

    print("")
    print("Optimizando regla temporal sobre VALIDACIÓN...")
    print(
        f"Eventos: {df['eventId'].astype(str).nunique()}"
    )
    print(f"Ventanas: {len(df)}")
    print(
        f"Umbrales candidatos: {len(thresholds)}"
    )

    for required_consecutive in CONSECUTIVE_OPTIONS:
        print(
            "  Analizando "
            f"k={required_consecutive} ventanas consecutivas..."
        )

        for threshold in thresholds:
            detail = event_predictions(
                df,
                float(threshold),
                required_consecutive,
            )

            rows.append(
                event_metrics(
                    detail,
                    float(threshold),
                    required_consecutive,
                )
            )

    sweep = pd.DataFrame(rows)

    best_youden = choose_best_youden(sweep)

    best_target = choose_target_sensitivity(
        sweep,
        MIN_SENSITIVITY_TARGET,
    )

    # Regla de referencia inspirada en el OSD original:
    # 3 ventanas consecutivas para alarma.
    k3_candidates = sweep[
        sweep["required_consecutive"] == 3
    ]

    best_k3_youden = choose_best_youden(
        k3_candidates
    )

    selected_rule = (
        best_target
        if best_target is not None
        else best_youden
    )

    selected_detail = event_predictions(
        df,
        float(selected_rule["threshold"]),
        int(selected_rule["required_consecutive"]),
    )

    sweep_path = (
        OUTPUT_DIR
        / "barrido_reglas_temporales.csv"
    )

    detail_path = (
        OUTPUT_DIR
        / "detalle_eventos_mejor_regla.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "reglas_candidatas_validacion.json"
    )

    summary_path = (
        OUTPUT_DIR
        / "resumen_regla_temporal_validacion.txt"
    )

    plot_path = (
        OUTPUT_DIR
        / "curva_sensibilidad_fpr_temporal.png"
    )

    sweep.to_csv(
        sweep_path,
        index=False,
    )

    selected_detail.to_csv(
        detail_path,
        index=False,
    )

    candidates = {
        "validation_only": True,
        "window_seconds": WINDOW_SECONDS,
        "minimum_sensitivity_target": (
            MIN_SENSITIVITY_TARGET
        ),
        "best_youden": row_to_json_dict(
            best_youden
        ),
        "best_target_sensitivity": (
            row_to_json_dict(best_target)
        ),
        "best_k3_youden": row_to_json_dict(
            best_k3_youden
        ),
        "selected_candidate": row_to_json_dict(
            selected_rule
        ),
    }

    json_path.write_text(
        json.dumps(
            candidates,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Curvas de sensibilidad-FPR por k.
    plt.figure()

    for required_consecutive in CONSECUTIVE_OPTIONS:
        subset = sweep[
            sweep["required_consecutive"]
            == required_consecutive
        ].sort_values("fpr")

        plt.plot(
            subset["fpr"],
            subset["sensitivity"],
            label=f"k={required_consecutive}",
        )

    plt.xlabel("Tasa de falsos positivos por evento")
    plt.ylabel("Sensibilidad por evento")
    plt.title(
        "Regla temporal sobre validación"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        plot_path,
        dpi=180,
    )
    plt.close()

    lines = [
        "OPTIMIZACIÓN DE REGLA TEMPORAL — VALIDACIÓN V2",
        "=" * 78,
        "",
        "Este análisis NO utiliza el conjunto TEST.",
        "",
        f"Entrada: {INPUT_PATH}",
        f"Eventos: {df['eventId'].astype(str).nunique()}",
        f"Ventanas de 5 s: {len(df)}",
        f"Objetivo mínimo de sensibilidad: "
        f"{MIN_SENSITIVITY_TARGET:.2f}",
        "",
        "MEJOR REGLA SEGÚN YOUDEN",
    ]

    for key, value in best_youden.to_dict().items():
        lines.append(f"{key}: {value}")

    lines.extend(
        [
            "",
            "MEJOR REGLA CON SENSIBILIDAD OBJETIVO",
        ]
    )

    if best_target is None:
        lines.append(
            "No se encontró ninguna regla con la "
            "sensibilidad objetivo."
        )
    else:
        for key, value in best_target.to_dict().items():
            lines.append(f"{key}: {value}")

    lines.extend(
        [
            "",
            "MEJOR REGLA k=3 SEGÚN YOUDEN",
            "(k=3 corresponde a tres ventanas consecutivas de 5 s)",
        ]
    )

    for key, value in best_k3_youden.to_dict().items():
        lines.append(f"{key}: {value}")

    lines.extend(
        [
            "",
            "REGLA CANDIDATA SELECCIONADA",
        ]
    )

    for key, value in selected_rule.to_dict().items():
        lines.append(f"{key}: {value}")

    lines.extend(
        [
            "",
            "INTERPRETACIÓN",
            (
                "El tiempo de alarma se mide desde la primera "
                "ventana conservada del evento, no necesariamente "
                "desde el inicio clínico exacto de la crisis."
            ),
            (
                "La regla todavía es candidata: debe fijarse antes "
                "de abrir TEST."
            ),
        ]
    )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("")
    print("=" * 78)
    print("OPTIMIZACIÓN TEMPORAL COMPLETADA")
    print("=" * 78)
    print(
        "Regla seleccionada: "
        f"umbral={float(selected_rule['threshold']):.6f}, "
        f"k={int(selected_rule['required_consecutive'])}"
    )
    print(
        "Sensibilidad: "
        f"{float(selected_rule['sensitivity']):.6f}"
    )
    print(
        "FPR: "
        f"{float(selected_rule['fpr']):.6f}"
    )
    print(
        "Especificidad: "
        f"{float(selected_rule['specificity']):.6f}"
    )
    print(f"Resumen: {summary_path}")
    print("")
    print("TEST todavía no fue utilizado.")


if __name__ == "__main__":
    main()

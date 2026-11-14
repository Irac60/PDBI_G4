"""
diagnosticar_balance_usuario_nocturno_v2.py
===========================================

Diagnóstico rápido del dataset nocturno V2 limpio.

Objetivo:
- comprobar si la clase (crisis/no crisis) está muy asociada a un usuario;
- ver cuántos eventos y ventanas aporta cada usuario en TRAIN/VALIDATION;
- revisar si las probabilidades del modelo cambian mucho por usuario;
- NO reentrenar;
- NO abrir TEST.

Entradas:
C:\\Users\\cesar\\osd\\dataset_nocturno_osdb_v1\\pipeline_nocturno_v2_limpio\\
    trainData.csv
    valData.csv
    baseline_fft_logreg_nocturno_v2_limpio\\probabilidades_validation.csv

Salida:
...\\diagnostico_balance_usuario\\resumen_balance_usuario.txt
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

TRAIN_PATH = PIPELINE_DIR / "trainData.csv"
VAL_PATH = PIPELINE_DIR / "valData.csv"

PROB_PATH = (
    PIPELINE_DIR
    / "baseline_fft_logreg_nocturno_v2_limpio"
    / "probabilidades_validation.csv"
)

OUTPUT_DIR = (
    PIPELINE_DIR
    / "diagnostico_balance_usuario"
)

SUMMARY_PATH = OUTPUT_DIR / "resumen_balance_usuario.txt"
TABLE_PATH = OUTPUT_DIR / "balance_usuario_split.csv"
VAL_SCORE_PATH = OUTPUT_DIR / "scores_validacion_por_usuario.csv"


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

    raise ValueError(
        f"Etiqueta no binaria: {value!r}"
    )


def normalize_event_id(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        head = text[:-2]

        if head.replace("-", "").isdigit():
            text = head

    return text


def event_table(
    path: Path,
    split_name: str,
) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        low_memory=False,
    )

    df = df.copy()

    df["label"] = (
        df["type"]
        .map(normalize_label)
    )

    df["event_id_norm"] = (
        df["eventId"]
        .map(normalize_event_id)
    )

    events = (
        df.groupby(
            "event_id_norm",
            sort=False,
        )
        .agg(
            eventId=("eventId", "first"),
            userId=("userId", "first"),
            label=("label", "first"),
            n_windows=("eventId", "size"),
        )
        .reset_index()
    )

    events["split"] = split_name

    return events


def balance_table(
    events: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (split_name, user_id), group in events.groupby(
        ["split", "userId"],
        dropna=False,
    ):
        positives = group.loc[
            group["label"] == 1
        ]

        negatives = group.loc[
            group["label"] == 0
        ]

        rows.append(
            {
                "split": split_name,
                "userId": user_id,
                "positive_events": len(positives),
                "negative_events": len(negatives),
                "positive_windows": int(
                    positives["n_windows"].sum()
                ),
                "negative_windows": int(
                    negatives["n_windows"].sum()
                ),
                "total_events": len(group),
                "total_windows": int(
                    group["n_windows"].sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        TRAIN_PATH,
        VAL_PATH,
        PROB_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró:\n{path}"
            )

    train_events = event_table(
        TRAIN_PATH,
        "train",
    )

    val_events = event_table(
        VAL_PATH,
        "validation",
    )

    all_events = pd.concat(
        [train_events, val_events],
        ignore_index=True,
    )

    balance = balance_table(
        all_events
    )

    balance.to_csv(
        TABLE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Probabilidades de validación por usuario y clase.
    # --------------------------------------------------------
    prob = pd.read_csv(
        PROB_PATH,
        low_memory=False,
    )

    if "probability_seizure" not in prob.columns:
        raise RuntimeError(
            "probabilidades_validation.csv no contiene "
            "probability_seizure."
        )

    prob["label_norm"] = (
        prob["label"]
        .map(normalize_label)
        if "label" in prob.columns
        else prob["type"].map(normalize_label)
    )

    prob["userId"] = (
        prob["userId"].astype(str)
    )

    score_summary = (
        prob.groupby(
            ["userId", "label_norm"],
            dropna=False,
        )
        .agg(
            windows=(
                "probability_seizure",
                "size",
            ),
            mean_probability=(
                "probability_seizure",
                "mean",
            ),
            median_probability=(
                "probability_seizure",
                "median",
            ),
            min_probability=(
                "probability_seizure",
                "min",
            ),
            max_probability=(
                "probability_seizure",
                "max",
            ),
        )
        .reset_index()
    )

    score_summary.to_csv(
        VAL_SCORE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Métrica simple de "pureza de clase por usuario".
    # Si un usuario aporta casi solo una clase, existe confusión fuerte.
    # --------------------------------------------------------
    train_balance = balance.loc[
        balance["split"] == "train"
    ].copy()

    warning_rows = []

    for _, row in train_balance.iterrows():
        total = (
            row["positive_events"]
            + row["negative_events"]
        )

        if total == 0:
            continue

        dominant = max(
            row["positive_events"],
            row["negative_events"],
        )

        purity = dominant / total

        warning_rows.append(
            {
                "userId": row["userId"],
                "class_purity_events": purity,
                "positive_events": row[
                    "positive_events"
                ],
                "negative_events": row[
                    "negative_events"
                ],
            }
        )

    warning_df = pd.DataFrame(
        warning_rows
    ).sort_values(
        by="class_purity_events",
        ascending=False,
    )

    severe_users = warning_df.loc[
        warning_df[
            "class_purity_events"
        ] >= 0.80
    ]

    # --------------------------------------------------------
    # Texto.
    # --------------------------------------------------------
    lines = [
        "DIAGNÓSTICO DE BALANCE POR USUARIO — NOCTURNO V2",
        "=" * 78,
        "",
        "Este diagnóstico usa TRAIN y VALIDATION.",
        "TEST NO se utiliza.",
        "",
        "BALANCE DE EVENTOS Y VENTANAS POR USUARIO",
        balance.to_string(index=False),
        "",
        "PUREZA DE CLASE POR USUARIO EN TRAIN",
        warning_df.to_string(index=False),
        "",
        (
            "Usuarios con >=80 % de sus eventos TRAIN en una sola clase: "
            f"{len(severe_users)}"
        ),
        "",
        "SCORES DEL MODELO EN VALIDATION POR USUARIO Y CLASE",
        score_summary.to_string(index=False),
        "",
        "INTERPRETACIÓN AUTOMÁTICA",
    ]

    if len(severe_users) > 0:
        lines.extend(
            [
                (
                    "Existe una asociación fuerte entre usuario y clase "
                    "en TRAIN. El modelo puede aprender identidad/dispositivo "
                    "en vez de movimiento de crisis."
                ),
                (
                    "Antes de probar modelos más complejos conviene "
                    "reconstruir el conjunto de entrenamiento con negativos "
                    "mejor emparejados por usuario."
                ),
            ]
        )
    else:
        lines.extend(
            [
                (
                    "No se detectó una asociación extrema usuario-clase "
                    "según este criterio."
                ),
                (
                    "El siguiente paso sería revisar diferencias de "
                    "dispositivo/escala y reducir características."
                ),
            ]
        )

    SUMMARY_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("")
    print("=" * 78)
    print("DIAGNÓSTICO COMPLETADO")
    print("=" * 78)
    print(
        f"Usuarios TRAIN con pureza de clase >= 0.80: "
        f"{len(severe_users)}"
    )
    print(f"Resumen: {SUMMARY_PATH}")
    print("")
    print("TEST permanece cerrado.")


if __name__ == "__main__":
    main()

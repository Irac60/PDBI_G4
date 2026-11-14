"""
analizar_hr_nocturno_osdb_v1.py
===============================

Analiza si la frecuencia cardiaca (HR) disponible en OSDB puede servir
como señal de corroboración para el WARNING de acelerometría k=2.

NO ajusta todavía ningún umbral de HR.
NO usa TEST.
Usa únicamente TRAIN + VALIDATION.

Entradas:
- JSON nocturnos extraídos.
- manifiesto de split nocturno V2 limpio.
- detalle k=2 de TRAIN y VALIDATION.

Salidas:
- cobertura_hr_eventos.csv
- cobertura_hr_por_split_clase.csv
- eventos_warning_k2_con_hr.csv
- resumen_hr_nocturno.txt

Ejecución:
    python .\\analizar_hr_nocturno_osdb_v1.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(
    r"C:\Users\cesar\osd\dataset_nocturno_osdb_v1"
)

EXTRACTED_DIR = ROOT / "dataset_extraido_v1"

POS_JSON = (
    EXTRACTED_DIR
    / "osdb_nocturno_tonic_clonic.json"
)

NEG_JSON = (
    EXTRACTED_DIR
    / "osdb_nocturno_false_alarms.json"
)

PIPELINE_DIR = (
    ROOT / "pipeline_nocturno_v2_limpio"
)

MANIFEST_PATH = (
    PIPELINE_DIR
    / "manifiesto_division_eventos_nocturno_v2_limpio.csv"
)

COMPARE_DIR = (
    PIPELINE_DIR
    / "comparacion_k2_k3_nocturno"
)

TRAIN_K2_PATH = (
    COMPARE_DIR / "detalle_train_k2.csv"
)

VAL_K2_PATH = (
    COMPARE_DIR / "detalle_validation_k2.csv"
)

OUTPUT_DIR = (
    PIPELINE_DIR / "analisis_hr_nocturno"
)


def normalize_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        candidate = text[:-2]

        if candidate.replace("-", "").isdigit():
            text = candidate

    return text


def numeric_if_valid(
    value: Any,
) -> float | None:
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(number):
        return None

    return number


def extract_hr_from_datapoint(
    datapoint: dict[str, Any],
) -> float | None:
    candidates = [
        "hr",
        "HR",
        "heartRate",
        "heart_rate",
        "heartrate",
    ]

    for key in candidates:
        if key not in datapoint:
            continue

        value = numeric_if_valid(
            datapoint.get(key)
        )

        if value is None:
            continue

        # Filtro fisiológico amplio; evita -1/0 usados como "sin dato".
        if 20 <= value <= 250:
            return value

    return None


def extract_spo2_from_datapoint(
    datapoint: dict[str, Any],
) -> float | None:
    candidates = [
        "O2",
        "o2",
        "SpO2",
        "spo2",
        "oxygen",
        "oxygenSaturation",
    ]

    for key in candidates:
        if key not in datapoint:
            continue

        value = numeric_if_valid(
            datapoint.get(key)
        )

        if value is None:
            continue

        if 50 <= value <= 100:
            return value

    return None


def event_signal_summary(
    event: dict[str, Any],
    source_class: str,
) -> dict[str, Any]:
    event_id = normalize_id(
        event.get("id")
    )

    datapoints = event.get(
        "datapoints",
        []
    )

    if not isinstance(datapoints, list):
        datapoints = []

    hr_values = []
    spo2_values = []

    for datapoint in datapoints:
        if not isinstance(
            datapoint,
            dict,
        ):
            continue

        hr = extract_hr_from_datapoint(
            datapoint
        )

        spo2 = extract_spo2_from_datapoint(
            datapoint
        )

        if hr is not None:
            hr_values.append(hr)

        if spo2 is not None:
            spo2_values.append(spo2)

    hr_array = np.asarray(
        hr_values,
        dtype=float,
    )

    spo2_array = np.asarray(
        spo2_values,
        dtype=float,
    )

    result = {
        "eventId": event_id,
        "source_class": source_class,
        "userId": event.get("userId"),
        "n_datapoints": len(datapoints),
        "hr_valid_count": len(hr_array),
        "hr_valid_fraction": (
            len(hr_array) / len(datapoints)
            if len(datapoints)
            else 0.0
        ),
        "spo2_valid_count": len(
            spo2_array
        ),
        "spo2_valid_fraction": (
            len(spo2_array)
            / len(datapoints)
            if len(datapoints)
            else 0.0
        ),
    }

    if len(hr_array):
        result.update(
            {
                "hr_min": float(
                    np.min(hr_array)
                ),
                "hr_max": float(
                    np.max(hr_array)
                ),
                "hr_mean": float(
                    np.mean(hr_array)
                ),
                "hr_median": float(
                    np.median(hr_array)
                ),
                "hr_std": float(
                    np.std(hr_array)
                ),
                "hr_range": float(
                    np.max(hr_array)
                    - np.min(hr_array)
                ),
                "hr_first": float(
                    hr_array[0]
                ),
                "hr_last": float(
                    hr_array[-1]
                ),
                "hr_delta_last_first": float(
                    hr_array[-1]
                    - hr_array[0]
                ),
            }
        )
    else:
        result.update(
            {
                "hr_min": np.nan,
                "hr_max": np.nan,
                "hr_mean": np.nan,
                "hr_median": np.nan,
                "hr_std": np.nan,
                "hr_range": np.nan,
                "hr_first": np.nan,
                "hr_last": np.nan,
                "hr_delta_last_first": np.nan,
            }
        )

    if len(spo2_array):
        result.update(
            {
                "spo2_min": float(
                    np.min(spo2_array)
                ),
                "spo2_max": float(
                    np.max(spo2_array)
                ),
                "spo2_mean": float(
                    np.mean(spo2_array)
                ),
            }
        )
    else:
        result.update(
            {
                "spo2_min": np.nan,
                "spo2_max": np.nan,
                "spo2_mean": np.nan,
            }
        )

    return result


def load_events(
    path: Path,
    source_class: str,
) -> list[dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    if not isinstance(
        data,
        list,
    ):
        raise RuntimeError(
            f"{path.name} no contiene una lista JSON."
        )

    return [
        event_signal_summary(
            event,
            source_class,
        )
        for event in data
        if isinstance(event, dict)
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        POS_JSON,
        NEG_JSON,
        MANIFEST_PATH,
        TRAIN_K2_PATH,
        VAL_K2_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró:\n{path}"
            )

    rows = []

    rows.extend(
        load_events(
            POS_JSON,
            "positive",
        )
    )

    rows.extend(
        load_events(
            NEG_JSON,
            "negative",
        )
    )

    coverage = pd.DataFrame(rows)

    manifest = pd.read_csv(
        MANIFEST_PATH,
        low_memory=False,
    )

    manifest["event_id_norm"] = (
        manifest["eventId"]
        .map(normalize_id)
    )

    coverage["event_id_norm"] = (
        coverage["eventId"]
        .map(normalize_id)
    )

    # Solo TRAIN + VALIDATION.
    split_table = (
        manifest[
            [
                "event_id_norm",
                "split",
                "label",
                "userId",
            ]
        ]
        .drop_duplicates(
            subset=["event_id_norm"]
        )
    )

    coverage = coverage.merge(
        split_table,
        on="event_id_norm",
        how="inner",
        suffixes=("", "_manifest"),
    )

    coverage = coverage.loc[
        coverage["split"].isin(
            ["train", "validation"]
        )
    ].copy()

    coverage_path = (
        OUTPUT_DIR
        / "cobertura_hr_eventos.csv"
    )

    coverage.to_csv(
        coverage_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_by_split = (
        coverage.groupby(
            ["split", "label"],
            dropna=False,
        )
        .agg(
            events=(
                "event_id_norm",
                "nunique",
            ),
            events_with_hr=(
                "hr_valid_count",
                lambda x: int(
                    (x > 0).sum()
                ),
            ),
            events_hr_fullish=(
                "hr_valid_fraction",
                lambda x: int(
                    (x >= 0.80).sum()
                ),
            ),
            median_hr_fraction=(
                "hr_valid_fraction",
                "median",
            ),
            median_hr_range=(
                "hr_range",
                "median",
            ),
            events_with_spo2=(
                "spo2_valid_count",
                lambda x: int(
                    (x > 0).sum()
                ),
            ),
        )
        .reset_index()
    )

    summary_split_path = (
        OUTPUT_DIR
        / "cobertura_hr_por_split_clase.csv"
    )

    summary_by_split.to_csv(
        summary_split_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Eventos que activan WARNING k=2.
    # --------------------------------------------------------
    train_k2 = pd.read_csv(
        TRAIN_K2_PATH,
        low_memory=False,
    )

    val_k2 = pd.read_csv(
        VAL_K2_PATH,
        low_memory=False,
    )

    train_k2["split"] = "train"
    val_k2["split"] = "validation"

    warning = pd.concat(
        [train_k2, val_k2],
        ignore_index=True,
    )

    warning["event_id_norm"] = (
        warning["eventId"]
        .map(normalize_id)
    )

    warning = warning.loc[
        warning["predicted_label"] == 1
    ].copy()

    warning_hr = warning.merge(
        coverage,
        on=[
            "event_id_norm",
            "split",
        ],
        how="left",
        suffixes=(
            "_warning",
            "_hr",
        ),
    )

    warning_path = (
        OUTPUT_DIR
        / "eventos_warning_k2_con_hr.csv"
    )

    warning_hr.to_csv(
        warning_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Resumen textual.
    # --------------------------------------------------------
    hr_events = int(
        (coverage["hr_valid_count"] > 0)
        .sum()
    )

    fullish_events = int(
        (
            coverage[
                "hr_valid_fraction"
            ]
            >= 0.80
        ).sum()
    )

    spo2_events = int(
        (coverage["spo2_valid_count"] > 0)
        .sum()
    )

    warning_with_hr = int(
        (
            warning_hr[
                "hr_valid_count"
            ]
            .fillna(0)
            > 0
        ).sum()
    )

    summary_path = (
        OUTPUT_DIR
        / "resumen_hr_nocturno.txt"
    )

    columns_for_warning = [
        column
        for column in [
            "split",
            "eventId_warning",
            "true_label",
            "predicted_label",
            "n_windows",
            "longest_positive_run",
            "hr_valid_count",
            "hr_valid_fraction",
            "hr_first",
            "hr_last",
            "hr_min",
            "hr_max",
            "hr_range",
            "spo2_valid_count",
        ]
        if column in warning_hr.columns
    ]

    lines = [
        "COBERTURA DE HR / SpO2 — DATASET NOCTURNO",
        "=" * 78,
        "",
        "Este análisis usa TRAIN + VALIDATION.",
        "TEST permanece cerrado.",
        "",
        f"Eventos analizados: {len(coverage)}",
        f"Eventos con al menos un HR válido: {hr_events}",
        (
            "Eventos con >=80 % de datapoints con HR válido: "
            f"{fullish_events}"
        ),
        f"Eventos con algún SpO2 válido: {spo2_events}",
        "",
        "COBERTURA POR SPLIT Y CLASE",
        summary_by_split.to_string(
            index=False
        ),
        "",
        "WARNING k=2",
        f"Eventos que activan WARNING: {len(warning)}",
        (
            "Eventos WARNING con al menos un HR válido: "
            f"{warning_with_hr}"
        ),
        "",
        "DETALLE WARNING + HR",
        (
            warning_hr[
                columns_for_warning
            ].to_string(
                index=False
            )
            if columns_for_warning
            else "No disponible."
        ),
        "",
        "DECISIÓN ORIENTATIVA",
    ]

    if (
        hr_events >= 0.70 * len(coverage)
        and warning_with_hr
        >= 0.70 * max(1, len(warning))
    ):
        lines.extend(
            [
                (
                    "La cobertura de HR parece suficiente para intentar "
                    "una regla de corroboración sobre TRAIN y evaluarla "
                    "después en VALIDATION."
                ),
                (
                    "El siguiente paso será estudiar cambio relativo de HR, "
                    "no un valor absoluto fijo."
                ),
            ]
        )
    else:
        lines.extend(
            [
                (
                    "La cobertura de HR de OSDB es insuficiente o demasiado "
                    "irregular para entrenar una corroboración fiable."
                ),
                (
                    "En ese caso conservaremos k=2 como WARNING y dejaremos "
                    "la corroboración para MAX30102/EEG en el prototipo."
                ),
            ]
        )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("")
    print("=" * 78)
    print("ANÁLISIS HR NOCTURNO COMPLETADO")
    print("=" * 78)
    print(
        f"Eventos TRAIN+VALIDATION: {len(coverage)}"
    )
    print(
        f"Con HR válido: {hr_events}"
    )
    print(
        f"Con >=80 % HR válido: {fullish_events}"
    )
    print(
        f"WARNING k=2 con HR: "
        f"{warning_with_hr}/{len(warning)}"
    )
    print(
        f"Con algún SpO2 válido: {spo2_events}"
    )
    print(f"Resumen: {summary_path}")
    print("")
    print("TEST permanece cerrado.")


if __name__ == "__main__":
    main()

"""
analizar_cambio_hr_warning_nocturno.py
======================================

Analiza HR de forma CAUSAL alrededor del WARNING k=2.

No usa el rango HR de todo el evento.
Para cada evento que activa WARNING:
1. Localiza las dos ventanas consecutivas positivas.
2. Identifica esas ventanas en los datapoints rawData originales.
3. Calcula una línea basal con hasta 3 HR válidos anteriores.
4. Calcula HR al momento del WARNING.
5. Calcula excursión de HR hasta 10 s después del WARNING.

Usa únicamente TRAIN + VALIDATION.
TEST permanece cerrado.

Salidas:
- warning_hr_causal_eventos.csv
- resumen_hr_causal_por_split_clase.csv
- resumen_hr_causal.txt
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

PIPELINE_DIR = (
    ROOT / "pipeline_nocturno_v2_limpio"
)

FEATURE_DIR = (
    PIPELINE_DIR
    / "baseline_fft_logreg_nocturno_v3_balance_usuario"
)

TRAIN_FEATURES = FEATURE_DIR / "features_train.csv"
VAL_FEATURES = FEATURE_DIR / "probabilidades_validation.csv"

TRAIN_DATA = PIPELINE_DIR / "trainData.csv"
VAL_DATA = PIPELINE_DIR / "valData.csv"

POS_JSON = (
    ROOT / "dataset_extraido_v1"
    / "osdb_nocturno_tonic_clonic.json"
)

NEG_JSON = (
    ROOT / "dataset_extraido_v1"
    / "osdb_nocturno_false_alarms.json"
)

OUTPUT_DIR = (
    PIPELINE_DIR
    / "analisis_hr_warning_causal"
)

POWER_THRESHOLD = 0.05425036036486526
RATIO_THRESHOLD = 0.436484300104836
K_WARNING = 2

M_COLS = [
    f"M{i:03d}"
    for i in range(125)
]


def normalize_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        candidate = text[:-2]

        if candidate.replace("-", "").isdigit():
            text = candidate

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

    raise ValueError(
        f"Etiqueta no binaria: {value!r}"
    )


def valid_hr(
    datapoint: dict[str, Any],
) -> float | None:
    for key in [
        "hr",
        "HR",
        "heartRate",
        "heart_rate",
        "heartrate",
    ]:
        if key not in datapoint:
            continue

        try:
            value = float(
                datapoint[key]
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if (
            np.isfinite(value)
            and 20 <= value <= 250
        ):
            return value

    return None


def load_event_dictionary() -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}

    for path in [
        POS_JSON,
        NEG_JSON,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró:\n{path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            events = json.load(handle)

        if not isinstance(
            events,
            list,
        ):
            raise RuntimeError(
                f"{path.name} no contiene lista."
            )

        for event in events:
            if not isinstance(
                event,
                dict,
            ):
                continue

            event_id = normalize_id(
                event.get("id")
            )

            if event_id:
                output[
                    event_id
                ] = event

    return output


def prepare_features(
    path: Path,
    split_name: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró:\n{path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    required = {
        "source_row",
        "eventId",
        "label",
        "power_3_8",
        "ratio_3_8",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise RuntimeError(
            f"{path.name}: faltan "
            + ", ".join(
                sorted(missing)
            )
        )

    out = df.copy()

    out[
        "event_id_norm"
    ] = out[
        "eventId"
    ].map(
        normalize_id
    )

    out[
        "label_norm"
    ] = out[
        "label"
    ].map(
        normalize_label
    )

    out[
        "window_positive"
    ] = (
        (
            out["power_3_8"]
            >= POWER_THRESHOLD
        )
        &
        (
            out["ratio_3_8"]
            >= RATIO_THRESHOLD
        )
    )

    out["split"] = split_name

    return out.sort_values(
        by=[
            "event_id_norm",
            "source_row",
        ]
    )


def prepare_raw_split(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró:\n{path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    missing = set(
        M_COLS
        + [
            "eventId",
        ]
    ) - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{path.name}: faltan "
            + ", ".join(
                sorted(missing)[:20]
            )
        )

    out = df.copy()

    # source_row fue creado a partir del índice original
    # del DataFrame al extraer características.
    out[
        "source_row"
    ] = np.arange(
        len(out),
        dtype=int,
    )

    out[
        "event_id_norm"
    ] = out[
        "eventId"
    ].map(
        normalize_id
    )

    return out


def first_warning_rows(
    group: pd.DataFrame,
) -> tuple[int, int] | None:
    """
    Devuelve source_row de:
    - primera ventana positiva de la pareja
    - segunda ventana positiva que dispara WARNING
    """
    current = 0
    run_start_source_row = None

    for _, row in group.iterrows():
        if bool(
            row["window_positive"]
        ):
            if current == 0:
                run_start_source_row = int(
                    row["source_row"]
                )

            current += 1

            if current >= K_WARNING:
                return (
                    int(
                        run_start_source_row
                    ),
                    int(
                        row["source_row"]
                    ),
                )
        else:
            current = 0
            run_start_source_row = None

    return None


def raw_vectors(
    event: dict[str, Any],
) -> list[tuple[int, np.ndarray, dict[str, Any]]]:
    datapoints = event.get(
        "datapoints",
        []
    )

    output = []

    if not isinstance(
        datapoints,
        list,
    ):
        return output

    for index, datapoint in enumerate(
        datapoints
    ):
        if not isinstance(
            datapoint,
            dict,
        ):
            continue

        raw = datapoint.get(
            "rawData"
        )

        if (
            not isinstance(
                raw,
                list,
            )
            or len(raw) < 125
        ):
            continue

        try:
            vector = np.asarray(
                raw[:125],
                dtype=float,
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if not np.isfinite(
            vector
        ).all():
            continue

        output.append(
            (
                index,
                vector,
                datapoint,
            )
        )

    return output


def map_flat_vector_to_raw_index(
    flat_vector: np.ndarray,
    candidates: list[
        tuple[
            int,
            np.ndarray,
            dict[str, Any],
        ]
    ],
) -> tuple[int | None, float]:
    """
    Busca la ventana rawData más parecida.
    Devuelve (índice original, max_abs_error).
    """
    best_index = None
    best_error = np.inf

    for raw_index, raw_vector, _ in candidates:
        if (
            raw_vector.shape
            != flat_vector.shape
        ):
            continue

        error = float(
            np.max(
                np.abs(
                    raw_vector
                    - flat_vector
                )
            )
        )

        if error < best_error:
            best_error = error
            best_index = raw_index

            if error == 0.0:
                break

    return (
        best_index,
        best_error,
    )


def hr_values_in_range(
    datapoints: list[Any],
    start_index: int,
    end_index: int,
) -> list[float]:
    output = []

    start_index = max(
        0,
        start_index,
    )

    end_index = min(
        len(datapoints) - 1,
        end_index,
    )

    for index in range(
        start_index,
        end_index + 1,
    ):
        datapoint = datapoints[
            index
        ]

        if not isinstance(
            datapoint,
            dict,
        ):
            continue

        hr = valid_hr(
            datapoint
        )

        if hr is not None:
            output.append(
                hr
            )

    return output


def last_hr_at_or_before(
    datapoints: list[Any],
    index: int,
) -> float | None:
    for candidate_index in range(
        min(
            index,
            len(datapoints) - 1,
        ),
        -1,
        -1,
    ):
        datapoint = datapoints[
            candidate_index
        ]

        if not isinstance(
            datapoint,
            dict,
        ):
            continue

        hr = valid_hr(
            datapoint
        )

        if hr is not None:
            return hr

    return None


def baseline_hr_before(
    datapoints: list[Any],
    index: int,
    max_samples: int = 3,
) -> tuple[
    float | None,
    int,
]:
    values = []

    for candidate_index in range(
        index - 1,
        -1,
        -1,
    ):
        datapoint = datapoints[
            candidate_index
        ]

        if not isinstance(
            datapoint,
            dict,
        ):
            continue

        hr = valid_hr(
            datapoint
        )

        if hr is None:
            continue

        values.append(
            hr
        )

        if len(
            values
        ) >= max_samples:
            break

    if not values:
        return None, 0

    return (
        float(
            np.median(
                values
            )
        ),
        len(values),
    )


def analyze_split(
    features: pd.DataFrame,
    raw_split: pd.DataFrame,
    events: dict[
        str,
        dict[str, Any],
    ],
) -> list[
    dict[str, Any]
]:
    results = []

    raw_lookup = raw_split.set_index(
        "source_row",
        drop=False,
    )

    for event_id, group in features.groupby(
        "event_id_norm",
        sort=False,
    ):
        warning_pair = first_warning_rows(
            group
        )

        if warning_pair is None:
            continue

        start_source_row, warning_source_row = (
            warning_pair
        )

        if (
            start_source_row
            not in raw_lookup.index
            or warning_source_row
            not in raw_lookup.index
        ):
            raise RuntimeError(
                f"source_row no encontrado para evento {event_id}"
            )

        start_row = raw_lookup.loc[
            start_source_row
        ]

        warning_row = raw_lookup.loc[
            warning_source_row
        ]

        if isinstance(
            start_row,
            pd.DataFrame,
        ):
            start_row = start_row.iloc[
                0
            ]

        if isinstance(
            warning_row,
            pd.DataFrame,
        ):
            warning_row = warning_row.iloc[
                0
            ]

        if normalize_id(
            start_row["eventId"]
        ) != event_id:
            raise RuntimeError(
                f"Mismatch eventId en start row: {event_id}"
            )

        if normalize_id(
            warning_row["eventId"]
        ) != event_id:
            raise RuntimeError(
                f"Mismatch eventId en warning row: {event_id}"
            )

        event = events.get(
            event_id
        )

        if event is None:
            raise RuntimeError(
                f"Evento {event_id} no está en JSON nocturno."
            )

        candidates = raw_vectors(
            event
        )

        start_vector = (
            start_row[
                M_COLS
            ].to_numpy(
                dtype=float
            )
        )

        warning_vector = (
            warning_row[
                M_COLS
            ].to_numpy(
                dtype=float
            )
        )

        start_raw_index, start_error = (
            map_flat_vector_to_raw_index(
                start_vector,
                candidates,
            )
        )

        warning_raw_index, warning_error = (
            map_flat_vector_to_raw_index(
                warning_vector,
                candidates,
            )
        )

        datapoints = event.get(
            "datapoints",
            []
        )

        if (
            start_raw_index is None
            or warning_raw_index is None
            or not isinstance(
                datapoints,
                list,
            )
        ):
            continue

        baseline, baseline_samples = (
            baseline_hr_before(
                datapoints,
                start_raw_index,
                max_samples=3,
            )
        )

        hr_at_warning = (
            last_hr_at_or_before(
                datapoints,
                warning_raw_index,
            )
        )

        # Hasta 10 s después: 2 datapoints de 5 s.
        horizon_end = (
            warning_raw_index
            + 2
        )

        hr_warning_to_plus10 = (
            hr_values_in_range(
                datapoints,
                start_raw_index,
                horizon_end,
            )
        )

        delta_warning = np.nan
        delta_warning_pct = np.nan
        max_increase_10s = np.nan
        max_abs_change_10s = np.nan
        hr_range_10s = np.nan

        if (
            baseline is not None
            and hr_at_warning is not None
        ):
            delta_warning = (
                hr_at_warning
                - baseline
            )

            if baseline != 0:
                delta_warning_pct = (
                    100.0
                    * delta_warning
                    / baseline
                )

        if (
            baseline is not None
            and hr_warning_to_plus10
        ):
            values = np.asarray(
                hr_warning_to_plus10,
                dtype=float,
            )

            max_increase_10s = float(
                np.max(
                    values
                    - baseline
                )
            )

            max_abs_change_10s = float(
                np.max(
                    np.abs(
                        values
                        - baseline
                    )
                )
            )

            hr_range_10s = float(
                np.max(
                    values
                )
                - np.min(
                    values
                )
            )

        label = int(
            group[
                "label_norm"
            ].iloc[
                0
            ]
        )

        split_name = str(
            group[
                "split"
            ].iloc[
                0
            ]
        )

        results.append(
            {
                "split": split_name,
                "eventId": event_id,
                "true_label": label,
                "start_source_row": (
                    start_source_row
                ),
                "warning_source_row": (
                    warning_source_row
                ),
                "start_raw_index": (
                    start_raw_index
                ),
                "warning_raw_index": (
                    warning_raw_index
                ),
                "start_match_max_abs_error": (
                    start_error
                ),
                "warning_match_max_abs_error": (
                    warning_error
                ),
                "baseline_hr": (
                    baseline
                ),
                "baseline_hr_samples": (
                    baseline_samples
                ),
                "hr_at_warning": (
                    hr_at_warning
                ),
                "delta_hr_warning_bpm": (
                    delta_warning
                ),
                "delta_hr_warning_pct": (
                    delta_warning_pct
                ),
                "max_hr_increase_through_plus10s_bpm": (
                    max_increase_10s
                ),
                "max_abs_hr_change_through_plus10s_bpm": (
                    max_abs_change_10s
                ),
                "hr_range_through_plus10s_bpm": (
                    hr_range_10s
                ),
                "hr_samples_from_run_start_through_plus10s": (
                    len(
                        hr_warning_to_plus10
                    )
                ),
            }
        )

    return results


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_features = prepare_features(
        TRAIN_FEATURES,
        "train",
    )

    val_features = prepare_features(
        VAL_FEATURES,
        "validation",
    )

    train_raw = prepare_raw_split(
        TRAIN_DATA
    )

    val_raw = prepare_raw_split(
        VAL_DATA
    )

    event_dict = load_event_dictionary()

    results = []

    results.extend(
        analyze_split(
            train_features,
            train_raw,
            event_dict,
        )
    )

    results.extend(
        analyze_split(
            val_features,
            val_raw,
            event_dict,
        )
    )

    events = pd.DataFrame(
        results
    )

    events_path = (
        OUTPUT_DIR
        / "warning_hr_causal_eventos.csv"
    )

    events.to_csv(
        events_path,
        index=False,
        encoding="utf-8-sig",
    )

    metric_columns = [
        "delta_hr_warning_bpm",
        "delta_hr_warning_pct",
        "max_hr_increase_through_plus10s_bpm",
        "max_abs_hr_change_through_plus10s_bpm",
        "hr_range_through_plus10s_bpm",
    ]

    summary_rows = []

    for (
        split_name,
        label,
    ), group in events.groupby(
        [
            "split",
            "true_label",
        ],
        dropna=False,
    ):
        row = {
            "split": split_name,
            "label": label,
            "warning_events": len(
                group
            ),
            "events_with_baseline_hr": int(
                group[
                    "baseline_hr"
                ].notna().sum()
            ),
            "events_with_hr_at_warning": int(
                group[
                    "hr_at_warning"
                ].notna().sum()
            ),
        }

        for metric in metric_columns:
            values = (
                group[
                    metric
                ].dropna()
            )

            row[
                f"{metric}_median"
            ] = (
                float(
                    values.median()
                )
                if not values.empty
                else np.nan
            )

            row[
                f"{metric}_min"
            ] = (
                float(
                    values.min()
                )
                if not values.empty
                else np.nan
            )

            row[
                f"{metric}_max"
            ] = (
                float(
                    values.max()
                )
                if not values.empty
                else np.nan
            )

        summary_rows.append(
            row
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_csv = (
        OUTPUT_DIR
        / "resumen_hr_causal_por_split_clase.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
    )

    mapping_ok = (
        (
            events[
                "start_match_max_abs_error"
            ]
            <= 1e-6
        )
        &
        (
            events[
                "warning_match_max_abs_error"
            ]
            <= 1e-6
        )
    )

    summary_txt = (
        OUTPUT_DIR
        / "resumen_hr_causal.txt"
    )

    detail_columns = [
        "split",
        "eventId",
        "true_label",
        "baseline_hr",
        "baseline_hr_samples",
        "hr_at_warning",
        "delta_hr_warning_bpm",
        "delta_hr_warning_pct",
        "max_hr_increase_through_plus10s_bpm",
        "max_abs_hr_change_through_plus10s_bpm",
        "hr_range_through_plus10s_bpm",
        "hr_samples_from_run_start_through_plus10s",
        "start_match_max_abs_error",
        "warning_match_max_abs_error",
    ]

    lines = [
        "HR CAUSAL ALREDEDOR DEL WARNING k=2",
        "=" * 78,
        "",
        "Usa únicamente TRAIN + VALIDATION.",
        "TEST permanece cerrado.",
        "",
        (
            "No se usa información HR posterior a 10 s "
            "después del WARNING."
        ),
        "",
        f"Eventos WARNING analizados: {len(events)}",
        (
            "Eventos con mapeo rawData exacto: "
            f"{int(mapping_ok.sum())}/{len(events)}"
        ),
        "",
        "RESUMEN POR SPLIT Y CLASE",
        summary_df.to_string(
            index=False
        ),
        "",
        "DETALLE POR EVENTO",
        (
            events[
                detail_columns
            ].to_string(
                index=False
            )
            if not events.empty
            else "Sin eventos."
        ),
        "",
        "SIGUIENTE DECISIÓN",
        (
            "Revisar si alguna métrica relativa de HR separa "
            "los WARNING verdaderos de los falsos en TRAIN "
            "sin perder las crisis de VALIDATION."
        ),
        (
            "Todavía NO se selecciona umbral HR y TEST "
            "permanece cerrado."
        ),
    ]

    summary_txt.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("")
    print("=" * 78)
    print("ANÁLISIS HR CAUSAL COMPLETADO")
    print("=" * 78)
    print(
        f"Eventos WARNING: {len(events)}"
    )
    print(
        "Mapeo exacto rawData: "
        f"{int(mapping_ok.sum())}/"
        f"{len(events)}"
    )
    print(
        f"Resumen: {summary_txt}"
    )
    print("")
    print("TEST permanece cerrado.")


if __name__ == "__main__":
    main()

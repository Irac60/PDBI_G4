"""
evaluar_test_nocturno_regla_fija.py
===================================

EVALUACIÓN FINAL del conjunto TEST nocturno usando una regla YA FIJADA.

NO optimiza nada.
NO prueba otros umbrales.
NO compara k=2 vs k=3.
NO cambia features.

Regla congelada:
- power_3_8 >= 0.05425036036486526
- ratio_3_8 >= 0.436484300104836
- k = 2 ventanas consecutivas

Entrada:
C:\\Users\\cesar\\osd\\dataset_nocturno_osdb_v1\\pipeline_nocturno_v2_limpio\\testData.csv

Salida:
...\\evaluacion_test_regla_fija\\

IMPORTANTE:
Una vez ejecutado, TEST queda abierto y sus resultados deben considerarse
evaluación final de esta versión del algoritmo.
"""

from __future__ import annotations

import json
import os
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import get_window


PIPELINE_DIR = Path(
    r"C:\Users\cesar\osd\dataset_nocturno_osdb_v1"
    r"\pipeline_nocturno_v2_limpio"
)

TEST_PATH = PIPELINE_DIR / "testData.csv"

OUTPUT_DIR = (
    PIPELINE_DIR
    / "evaluacion_test_regla_fija"
)

FS_HZ = 25.0
N_SAMPLES = 125

POWER_THRESHOLD = 0.05425036036486526
RATIO_THRESHOLD = 0.436484300104836
REQUIRED_CONSECUTIVE = 2

M_COLS = [
    f"M{i:03d}"
    for i in range(N_SAMPLES)
]


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

    raise ValueError(
        f"Etiqueta no binaria: {value!r}"
    )


def band_power(
    power: np.ndarray,
    freqs: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    mask = (
        (freqs >= low_hz)
        &
        (freqs < high_hz)
    )

    if not np.any(mask):
        return 0.0

    return float(
        power[mask].sum()
    )


def extract_osd_features(
    magnitude_mg: np.ndarray,
) -> tuple[float, float]:
    """
    Reproduce exactamente la definición usada en el baseline:
    - OSDB rawData ~ mg
    - conversión a g
    - eliminación de DC
    - ventana Hann
    - FFT
    - power_3_8
    - ratio_3_8 respecto a potencia total 0.2–12.5 Hz
    """
    magnitude_g = (
        np.asarray(
            magnitude_mg,
            dtype=np.float64,
        )
        / 1000.0
    )

    if magnitude_g.size != N_SAMPLES:
        raise ValueError(
            f"Se esperaban {N_SAMPLES} muestras."
        )

    if not np.isfinite(
        magnitude_g
    ).all():
        raise ValueError(
            "Ventana con valores no finitos."
        )

    dynamic = (
        magnitude_g
        - np.mean(magnitude_g)
    )

    hann = get_window(
        "hann",
        N_SAMPLES,
        fftbins=True,
    )

    spectrum = np.fft.rfft(
        dynamic * hann
    )

    power = (
        np.abs(spectrum) ** 2
    )

    freqs = np.fft.rfftfreq(
        N_SAMPLES,
        d=1.0 / FS_HZ,
    )

    total_power = band_power(
        power,
        freqs,
        0.2,
        FS_HZ / 2 + 1e-9,
    )

    power_3_8 = band_power(
        power,
        freqs,
        3.0,
        8.0,
    )

    ratio_3_8 = (
        power_3_8
        / (total_power + 1e-12)
    )

    return (
        float(power_3_8),
        float(ratio_3_8),
    )


def longest_true_run(
    values: np.ndarray,
) -> int:
    best = 0
    current = 0

    for value in values.astype(bool):
        if value:
            current += 1
            best = max(
                best,
                current,
            )
        else:
            current = 0

    return best


def first_alarm_window(
    values: np.ndarray,
    k: int,
) -> int | None:
    current = 0

    for index, value in enumerate(
        values.astype(bool),
        start=1,
    ):
        if value:
            current += 1
        else:
            current = 0

        if current >= k:
            return index

    return None


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró:\n{TEST_PATH}"
        )

    test = pd.read_csv(
        TEST_PATH,
        low_memory=False,
    )

    required = set(
        M_COLS
        + [
            "eventId",
            "userId",
            "type",
            "typeStr",
        ]
    )

    missing = required - set(
        test.columns
    )

    if missing:
        raise RuntimeError(
            "Faltan columnas en TEST:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    test = test.copy()

    test["source_row"] = np.arange(
        len(test),
        dtype=int,
    )

    test["event_id_norm"] = (
        test["eventId"]
        .map(normalize_event_id)
    )

    test["label_norm"] = (
        test["type"]
        .map(normalize_label)
    )

    # --------------------------------------------------------
    # Características OSD por ventana.
    # --------------------------------------------------------
    feature_rows = []

    for row_index, row in test.iterrows():
        magnitude = (
            row[M_COLS]
            .to_numpy(
                dtype=np.float64
            )
        )

        power_3_8, ratio_3_8 = (
            extract_osd_features(
                magnitude
            )
        )

        window_positive = (
            power_3_8
            >= POWER_THRESHOLD
            and
            ratio_3_8
            >= RATIO_THRESHOLD
        )

        feature_rows.append(
            {
                "source_row": int(
                    row_index
                ),
                "eventId": row[
                    "eventId"
                ],
                "event_id_norm": row[
                    "event_id_norm"
                ],
                "userId": row[
                    "userId"
                ],
                "typeStr": row[
                    "typeStr"
                ],
                "label": int(
                    row[
                        "label_norm"
                    ]
                ),
                "power_3_8": (
                    power_3_8
                ),
                "ratio_3_8": (
                    ratio_3_8
                ),
                "window_positive": int(
                    window_positive
                ),
            }
        )

    windows = pd.DataFrame(
        feature_rows
    )

    windows_path = (
        OUTPUT_DIR
        / "ventanas_test_regla_fija.csv"
    )

    windows.to_csv(
        windows_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Evaluación por evento.
    # --------------------------------------------------------
    event_rows = []

    for event_id, group in windows.groupby(
        "event_id_norm",
        sort=False,
    ):
        group = group.sort_values(
            by="source_row"
        )

        positives = (
            group[
                "window_positive"
            ]
            .to_numpy(
                dtype=bool
            )
        )

        longest_run = (
            longest_true_run(
                positives
            )
        )

        alarm_window = (
            first_alarm_window(
                positives,
                REQUIRED_CONSECUTIVE,
            )
        )

        prediction = int(
            longest_run
            >= REQUIRED_CONSECUTIVE
        )

        true_label = int(
            group[
                "label"
            ].iloc[
                0
            ]
        )

        event_rows.append(
            {
                "eventId": event_id,
                "userId": group[
                    "userId"
                ].iloc[
                    0
                ],
                "typeStr": group[
                    "typeStr"
                ].iloc[
                    0
                ],
                "true_label": (
                    true_label
                ),
                "predicted_label": (
                    prediction
                ),
                "n_windows": len(
                    group
                ),
                "positive_windows": int(
                    positives.sum()
                ),
                "longest_positive_run": (
                    longest_run
                ),
                "alarm_time_from_first_window_s": (
                    alarm_window * 5
                    if alarm_window
                    is not None
                    else np.nan
                ),
                "mean_power_3_8": float(
                    group[
                        "power_3_8"
                    ].mean()
                ),
                "max_power_3_8": float(
                    group[
                        "power_3_8"
                    ].max()
                ),
                "mean_ratio_3_8": float(
                    group[
                        "ratio_3_8"
                    ].mean()
                ),
                "max_ratio_3_8": float(
                    group[
                        "ratio_3_8"
                    ].max()
                ),
            }
        )

    events = pd.DataFrame(
        event_rows
    )

    event_path = (
        OUTPUT_DIR
        / "detalle_eventos_test_regla_fija.csv"
    )

    events.to_csv(
        event_path,
        index=False,
        encoding="utf-8-sig",
    )

    y = events[
        "true_label"
    ].to_numpy()

    p = events[
        "predicted_label"
    ].to_numpy()

    tp = int(
        ((y == 1) & (p == 1))
        .sum()
    )

    tn = int(
        ((y == 0) & (p == 0))
        .sum()
    )

    fp = int(
        ((y == 0) & (p == 1))
        .sum()
    )

    fn = int(
        ((y == 1) & (p == 0))
        .sum()
    )

    sensitivity = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp
        else 0.0
    )

    fpr = (
        1.0
        - specificity
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    balanced_accuracy = (
        sensitivity
        + specificity
    ) / 2.0

    detected_times = (
        events.loc[
            (
                events[
                    "true_label"
                ]
                == 1
            )
            &
            (
                events[
                    "predicted_label"
                ]
                == 1
            ),
            "alarm_time_from_first_window_s",
        ]
        .dropna()
    )

    median_alarm_time = (
        float(
            detected_times.median()
        )
        if not detected_times.empty
        else np.nan
    )

    # --------------------------------------------------------
    # Gráfica simple por evento.
    # --------------------------------------------------------
    plot_table = events.copy()

    plot_table["resultado"] = np.select(
        [
            (
                plot_table[
                    "true_label"
                ] == 1
            )
            &
            (
                plot_table[
                    "predicted_label"
                ] == 1
            ),
            (
                plot_table[
                    "true_label"
                ] == 0
            )
            &
            (
                plot_table[
                    "predicted_label"
                ] == 0
            ),
            (
                plot_table[
                    "true_label"
                ] == 0
            )
            &
            (
                plot_table[
                    "predicted_label"
                ] == 1
            ),
            (
                plot_table[
                    "true_label"
                ] == 1
            )
            &
            (
                plot_table[
                    "predicted_label"
                ] == 0
            ),
        ],
        [
            "TP",
            "TN",
            "FP",
            "FN",
        ],
        default="?",
    )

    plt.figure(
        figsize=(9, 4.5)
    )

    x = np.arange(
        len(plot_table)
    )

    plt.bar(
        x,
        plot_table[
            "longest_positive_run"
        ].to_numpy(),
    )

    plt.axhline(
        REQUIRED_CONSECUTIVE,
        linestyle="--",
        label=(
            f"k={REQUIRED_CONSECUTIVE}"
        ),
    )

    plt.xticks(
        x,
        [
            f"{eid}\n{res}"
            for eid, res in zip(
                plot_table[
                    "eventId"
                ],
                plot_table[
                    "resultado"
                ],
            )
        ],
        rotation=45,
        ha="right",
    )

    plt.ylabel(
        "Máxima secuencia de ventanas positivas"
    )

    plt.title(
        "TEST nocturno — regla fija"
    )

    plt.grid(
        True,
        axis="y",
    )

    plt.legend()
    plt.tight_layout()

    plot_path = (
        OUTPUT_DIR
        / "resultado_test_por_evento.png"
    )

    plt.savefig(
        plot_path,
        dpi=180,
    )

    plt.close()

    # --------------------------------------------------------
    # Resumen.
    # --------------------------------------------------------
    result_payload = {
        "locked_rule": {
            "power_threshold": (
                POWER_THRESHOLD
            ),
            "ratio_threshold": (
                RATIO_THRESHOLD
            ),
            "required_consecutive": (
                REQUIRED_CONSECUTIVE
            ),
        },
        "test_metrics": {
            "events": int(
                len(events)
            ),
            "positive_events": int(
                (y == 1).sum()
            ),
            "negative_events": int(
                (y == 0).sum()
            ),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "sensitivity": (
                sensitivity
            ),
            "specificity": (
                specificity
            ),
            "fpr": fpr,
            "precision": (
                precision
            ),
            "balanced_accuracy": (
                balanced_accuracy
            ),
            "median_alarm_time_s": (
                median_alarm_time
            ),
        },
    }

    json_path = (
        OUTPUT_DIR
        / "resultado_test_regla_fija.json"
    )

    json_path.write_text(
        json.dumps(
            result_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary_path = (
        OUTPUT_DIR
        / "resumen_test_regla_fija.txt"
    )

    lines = [
        "EVALUACIÓN TEST — REGLA NOCTURNA FIJA",
        "=" * 78,
        "",
        "IMPORTANTE",
        (
            "Este resultado abre el conjunto TEST. "
            "Los parámetros NO se modificaron usando TEST."
        ),
        "",
        "REGLA FIJA",
        f"power_threshold: {POWER_THRESHOLD}",
        f"ratio_threshold: {RATIO_THRESHOLD}",
        f"required_consecutive: {REQUIRED_CONSECUTIVE}",
        "",
        "RESULTADOS TEST",
        f"Eventos: {len(events)}",
        f"Positivos: {int((y == 1).sum())}",
        f"Negativos: {int((y == 0).sum())}",
        f"TP: {tp}",
        f"TN: {tn}",
        f"FP: {fp}",
        f"FN: {fn}",
        f"Sensibilidad: {sensitivity}",
        f"Especificidad: {specificity}",
        f"FPR: {fpr}",
        f"Precisión: {precision}",
        f"Balanced accuracy: {balanced_accuracy}",
        f"Mediana tiempo de alarma detectadas (s): {median_alarm_time}",
        "",
        "DETALLE POR EVENTO",
        events.to_string(
            index=False
        ),
        "",
        "INTERPRETACIÓN",
    ]

    if (
        sensitivity >= 0.80
        and specificity >= 0.75
    ):
        lines.extend(
            [
                (
                    "La regla mantiene un comportamiento prometedor "
                    "en el TEST nocturno reservado."
                ),
                (
                    "Puede congelarse como detector de movimiento "
                    "sospechoso para el prototipo, no como diagnóstico clínico."
                ),
            ]
        )
    else:
        lines.extend(
            [
                (
                    "La regla no mantiene simultáneamente sensibilidad "
                    "y especificidad suficientes en este TEST pequeño."
                ),
                (
                    "No se deben reajustar estos mismos parámetros mirando "
                    "TEST; cualquier nueva versión necesitará una nueva "
                    "evaluación independiente."
                ),
            ]
        )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("")
    print("=" * 78)
    print("TEST NOCTURNO EVALUADO — REGLA FIJA")
    print("=" * 78)
    print(f"TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    print(
        f"Sensibilidad={sensitivity:.3f}"
    )
    print(
        f"Especificidad={specificity:.3f}"
    )
    print(
        f"FPR={fpr:.3f}"
    )
    print(
        f"Balanced accuracy={balanced_accuracy:.3f}"
    )
    print(
        f"Mediana alarma={median_alarm_time} s"
    )
    print("")
    print(f"Resumen: {summary_path}")
    print("")
    print(
        "TEST ya fue abierto. No reajustar parámetros "
        "con estos resultados."
    )


if __name__ == "__main__":
    main()

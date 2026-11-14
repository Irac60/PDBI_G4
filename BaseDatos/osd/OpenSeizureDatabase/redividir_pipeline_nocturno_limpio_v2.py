"""
redividir_pipeline_nocturno_limpio_v2.py
========================================

Rehace SOLO la división TRAIN / VALIDATION / TEST del dataset nocturno.

Regla:
- Los 5 positivos con nota de incertidumbre se fuerzan a TRAIN.
- VALIDATION y TEST contienen únicamente crisis nocturnas "limpias".
- Los negativos nocturnos se reparten aproximadamente 10/4/4.
- No se vuelve a aplanar.
- No se entrena.
- TEST sigue cerrado.

Resultado esperado aproximado:
TRAIN: 11 positivos + 10 negativos = 21 eventos
VALIDATION: 3 positivos + 4 negativos = 7 eventos
TEST: 3 positivos + 4 negativos = 7 eventos

Entrada:
C:\\Users\\cesar\\osd\\dataset_nocturno_osdb_v1\\pipeline_nocturno_v1\\allData.csv

Salida:
C:\\Users\\cesar\\osd\\dataset_nocturno_osdb_v1\\pipeline_nocturno_v2_limpio\\
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SOURCE_DIR = Path(
    r"C:\Users\cesar\osd\dataset_nocturno_osdb_v1"
    r"\pipeline_nocturno_v1"
)

SOURCE_CSV = SOURCE_DIR / "allData.csv"

OUTPUT_DIR = Path(
    r"C:\Users\cesar\osd\dataset_nocturno_osdb_v1"
    r"\pipeline_nocturno_v2_limpio"
)

TRAIN_CSV = OUTPUT_DIR / "trainData.csv"
VAL_CSV = OUTPUT_DIR / "valData.csv"
TEST_CSV = OUTPUT_DIR / "testData.csv"
MANIFEST_CSV = (
    OUTPUT_DIR / "manifiesto_division_eventos_nocturno_v2_limpio.csv"
)
SUMMARY_TXT = (
    OUTPUT_DIR / "resumen_pipeline_nocturno_v2_limpio.txt"
)

RANDOM_SEED = 20261114

UNCERTAIN_POSITIVE_IDS = {
    "1333417",
    "1343957",
    "1343962",
    "1343971",
    "1354976",
}


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


def event_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    work["event_id_norm"] = (
        work["eventId"]
        .map(normalize_event_id)
    )

    work["_label"] = (
        work["type"]
        .map(normalize_label)
    )

    table = (
        work.groupby(
            "event_id_norm",
            sort=False,
        )
        .agg(
            eventId=("eventId", "first"),
            userId=("userId", "first"),
            typeStr=("typeStr", "first"),
            label=("_label", "first"),
            n_rows=("eventId", "size"),
        )
        .reset_index()
    )

    table["uncertain_positive"] = (
        table["event_id_norm"]
        .isin(UNCERTAIN_POSITIVE_IDS)
    )

    return table


def save_split(
    all_df: pd.DataFrame,
    event_ids: set[str],
    path: Path,
) -> pd.DataFrame:
    work = all_df.copy()

    work["event_id_norm"] = (
        work["eventId"]
        .map(normalize_event_id)
    )

    subset = work.loc[
        work["event_id_norm"]
        .isin(event_ids)
    ].copy()

    subset = subset.drop(
        columns=["event_id_norm"],
        errors="ignore",
    )

    subset.to_csv(
        path,
        index=False,
    )

    return subset


def split_metrics(
    path: Path,
) -> dict[str, Any]:
    df = pd.read_csv(
        path,
        low_memory=False,
    )

    df["event_id_norm"] = (
        df["eventId"]
        .map(normalize_event_id)
    )

    df["_label"] = (
        df["type"]
        .map(normalize_label)
    )

    events = (
        df.groupby("event_id_norm")
        .agg(
            userId=("userId", "first"),
            label=("_label", "first"),
            n_rows=("eventId", "size"),
        )
        .reset_index()
    )

    return {
        "rows": len(df),
        "events": len(events),
        "positive_events": int(
            (events["label"] == 1).sum()
        ),
        "negative_events": int(
            (events["label"] == 0).sum()
        ),
        "positive_rows": int(
            (df["_label"] == 1).sum()
        ),
        "negative_rows": int(
            (df["_label"] == 0).sum()
        ),
        "users": int(
            df["userId"]
            .astype(str)
            .nunique()
        ),
        "event_set": set(
            events["event_id_norm"]
        ),
        "user_set": set(
            df["userId"]
            .astype(str)
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not SOURCE_CSV.exists():
        raise FileNotFoundError(
            f"No se encontró:\n{SOURCE_CSV}"
        )

    all_df = pd.read_csv(
        SOURCE_CSV,
        low_memory=False,
    )

    events = event_table(
        all_df
    )

    positives = events.loc[
        events["label"] == 1
    ].copy()

    negatives = events.loc[
        events["label"] == 0
    ].copy()

    uncertain = positives.loc[
        positives["uncertain_positive"]
    ].copy()

    clean_positive = positives.loc[
        ~positives["uncertain_positive"]
    ].copy()

    if len(uncertain) != 5:
        raise RuntimeError(
            f"Esperaba 5 positivos dudosos; encontré {len(uncertain)}."
        )

    if len(clean_positive) != 12:
        raise RuntimeError(
            f"Esperaba 12 positivos limpios; encontré {len(clean_positive)}."
        )

    if len(negatives) != 18:
        raise RuntimeError(
            f"Esperaba 18 negativos; encontré {len(negatives)}."
        )

    # --------------------------------------------------------
    # Positivos limpios:
    # 6 train, 3 validation, 3 test.
    # --------------------------------------------------------
    clean_ids = clean_positive[
        "event_id_norm"
    ].to_numpy()

    clean_train_ids, clean_temp_ids = (
        train_test_split(
            clean_ids,
            test_size=0.50,
            random_state=RANDOM_SEED,
        )
    )

    clean_val_ids, clean_test_ids = (
        train_test_split(
            clean_temp_ids,
            test_size=0.50,
            random_state=RANDOM_SEED + 1,
        )
    )

    # --------------------------------------------------------
    # Negativos:
    # 10 train, 4 validation, 4 test.
    # --------------------------------------------------------
    negative_ids = negatives[
        "event_id_norm"
    ].to_numpy()

    neg_train_ids, neg_temp_ids = (
        train_test_split(
            negative_ids,
            test_size=8,
            random_state=RANDOM_SEED + 2,
        )
    )

    neg_val_ids, neg_test_ids = (
        train_test_split(
            neg_temp_ids,
            test_size=4,
            random_state=RANDOM_SEED + 3,
        )
    )

    uncertain_ids = set(
        uncertain["event_id_norm"]
    )

    train_ids = (
        set(clean_train_ids)
        | uncertain_ids
        | set(neg_train_ids)
    )

    val_ids = (
        set(clean_val_ids)
        | set(neg_val_ids)
    )

    test_ids = (
        set(clean_test_ids)
        | set(neg_test_ids)
    )

    # --------------------------------------------------------
    # Verificaciones.
    # --------------------------------------------------------
    if train_ids & val_ids:
        raise RuntimeError(
            "Fuga TRAIN-VALIDATION."
        )

    if train_ids & test_ids:
        raise RuntimeError(
            "Fuga TRAIN-TEST."
        )

    if val_ids & test_ids:
        raise RuntimeError(
            "Fuga VALIDATION-TEST."
        )

    if uncertain_ids & val_ids:
        raise RuntimeError(
            "Hay positivos dudosos en VALIDATION."
        )

    if uncertain_ids & test_ids:
        raise RuntimeError(
            "Hay positivos dudosos en TEST."
        )

    all_event_ids = set(
        events["event_id_norm"]
    )

    recovered = (
        train_ids
        | val_ids
        | test_ids
    )

    if recovered != all_event_ids:
        missing = (
            all_event_ids - recovered
        )
        extra = (
            recovered - all_event_ids
        )

        raise RuntimeError(
            "La nueva división no recupera todos los eventos. "
            f"Faltantes={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    # --------------------------------------------------------
    # Guardar splits.
    # --------------------------------------------------------
    save_split(
        all_df,
        train_ids,
        TRAIN_CSV,
    )

    save_split(
        all_df,
        val_ids,
        VAL_CSV,
    )

    save_split(
        all_df,
        test_ids,
        TEST_CSV,
    )

    # --------------------------------------------------------
    # Manifiesto.
    # --------------------------------------------------------
    parts = []

    for split_name, ids in [
        ("train", train_ids),
        ("validation", val_ids),
        ("test", test_ids),
    ]:
        part = events.loc[
            events["event_id_norm"]
            .isin(ids)
        ].copy()

        part["split"] = split_name
        parts.append(part)

    manifest = pd.concat(
        parts,
        ignore_index=True,
    )

    manifest.to_csv(
        MANIFEST_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Métricas.
    # --------------------------------------------------------
    train_m = split_metrics(
        TRAIN_CSV
    )

    val_m = split_metrics(
        VAL_CSV
    )

    test_m = split_metrics(
        TEST_CSV
    )

    user_overlap_train_val = (
        train_m["user_set"]
        & val_m["user_set"]
    )

    user_overlap_train_test = (
        train_m["user_set"]
        & test_m["user_set"]
    )

    user_overlap_val_test = (
        val_m["user_set"]
        & test_m["user_set"]
    )

    # Confirmar dudas por split.
    uncertain_manifest = manifest.loc[
        manifest["event_id_norm"]
        .isin(UNCERTAIN_POSITIVE_IDS)
    ][
        [
            "split",
            "eventId",
            "userId",
            "typeStr",
            "n_rows",
        ]
    ].copy()

    lines = [
        "PIPELINE NOCTURNO OSDB V2 — SPLIT LIMPIO",
        "=" * 78,
        "",
        "REGLA",
        (
            "Los 5 positivos con nota de incertidumbre se fuerzan "
            "exclusivamente a TRAIN."
        ),
        (
            "VALIDATION y TEST contienen solo crisis Tonic-Clonic "
            "nocturnas sin nota de incertidumbre."
        ),
        "",
        "EVENTOS DISPONIBLES",
        f"Positivos totales: {len(positives)}",
        f"Positivos limpios: {len(clean_positive)}",
        f"Positivos dudosos: {len(uncertain)}",
        f"Negativos nocturnos: {len(negatives)}",
        "",
        "TRAIN",
        f"Filas: {train_m['rows']}",
        f"Eventos: {train_m['events']}",
        f"Positivos: {train_m['positive_events']}",
        f"Negativos: {train_m['negative_events']}",
        f"Filas positivas: {train_m['positive_rows']}",
        f"Filas negativas: {train_m['negative_rows']}",
        f"Usuarios: {train_m['users']}",
        "",
        "VALIDATION",
        f"Filas: {val_m['rows']}",
        f"Eventos: {val_m['events']}",
        f"Positivos: {val_m['positive_events']}",
        f"Negativos: {val_m['negative_events']}",
        f"Filas positivas: {val_m['positive_rows']}",
        f"Filas negativas: {val_m['negative_rows']}",
        f"Usuarios: {val_m['users']}",
        "",
        "TEST",
        f"Filas: {test_m['rows']}",
        f"Eventos: {test_m['events']}",
        f"Positivos: {test_m['positive_events']}",
        f"Negativos: {test_m['negative_events']}",
        f"Filas positivas: {test_m['positive_rows']}",
        f"Filas negativas: {test_m['negative_rows']}",
        f"Usuarios: {test_m['users']}",
        "",
        "POSITIVOS DUDOSOS",
        uncertain_manifest.to_string(
            index=False
        ),
        "",
        "VERIFICACIONES",
        "Positivos dudosos en VALIDATION: 0",
        "Positivos dudosos en TEST: 0",
        "Train ∩ Validation eventId: 0",
        "Train ∩ Test eventId: 0",
        "Validation ∩ Test eventId: 0",
        "",
        "SUPERPOSICIÓN DE USUARIOS",
        f"Train ∩ Validation: {len(user_overlap_train_val)}",
        f"Train ∩ Test: {len(user_overlap_train_test)}",
        f"Validation ∩ Test: {len(user_overlap_val_test)}",
        "",
        "SIGUIENTE PASO",
        (
            "Entrenar el mismo baseline FFT + regresión logística "
            "con TRAIN y VALIDATION de esta nueva división."
        ),
        (
            "TEST permanece cerrado."
        ),
    ]

    SUMMARY_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("")
    print("=" * 78)
    print("REDIVISIÓN NOCTURNA V2 COMPLETADA")
    print("=" * 78)

    print(
        f"TRAIN: {train_m['events']} eventos "
        f"({train_m['positive_events']}+ / "
        f"{train_m['negative_events']}-)"
    )

    print(
        f"VALIDATION: {val_m['events']} eventos "
        f"({val_m['positive_events']}+ / "
        f"{val_m['negative_events']}-)"
    )

    print(
        f"TEST: {test_m['events']} eventos "
        f"({test_m['positive_events']}+ / "
        f"{test_m['negative_events']}-)"
    )

    print(
        "Positivos dudosos en validation/test: 0"
    )
    print(
        "Fuga de eventId: 0"
    )
    print(f"Resumen: {SUMMARY_TXT}")
    print("")
    print("TEST permanece cerrado.")


if __name__ == "__main__":
    main()

"""
preparar_pipeline_nocturno_osdb_v1.py
=====================================

Prepara el pipeline nocturno V1 a partir de los 17 eventos Tonic-Clonic
nocturnos y 18 False Alarm de sueño/reposo ya extraídos.

Hace:
1. Une positivos y negativos en allData.json.
2. Normaliza seizureTimes de los positivos.
3. Aplana a ventanas de 5 s / 125 muestras con la herramienta oficial.
4. Divide por EVENTO en TRAIN / VALIDATION / TEST (~60/20/20),
   estratificando por clase.
5. Comprueba fuga de eventId.
6. Guarda un resumen.

NO entrena.
NO usa TEST para ajustar nada.

Nota:
El split es por EVENTO, no por usuario, porque solo hay 3 usuarios con
crisis nocturnas. La superposición de userId se reporta explícitamente.

Ejecución:
    python .\preparar_pipeline_nocturno_osdb_v1.py
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from user_tools.nnTraining2 import flattenData


# ============================================================
# RUTAS
# ============================================================

REPO_DIR = Path(r"C:\Users\cesar\osd\OpenSeizureDatabase")

SOURCE_DIR = Path(
    r"C:\Users\cesar\osd\dataset_nocturno_osdb_v1"
    r"\dataset_extraido_v1"
)

POSITIVE_JSON = (
    SOURCE_DIR / "osdb_nocturno_tonic_clonic.json"
)

NEGATIVE_JSON = (
    SOURCE_DIR / "osdb_nocturno_false_alarms.json"
)

BASE_CONFIG_PATH = (
    REPO_DIR / "nnConfig_prototipo_v2.json"
)

PIPELINE_DIR = Path(
    r"C:\Users\cesar\osd\dataset_nocturno_osdb_v1"
    r"\pipeline_nocturno_v1"
)

ALL_JSON = PIPELINE_DIR / "allData.json"
ALL_CSV = PIPELINE_DIR / "allData.csv"
TRAIN_CSV = PIPELINE_DIR / "trainData.csv"
VAL_CSV = PIPELINE_DIR / "valData.csv"
TEST_CSV = PIPELINE_DIR / "testData.csv"

SPLIT_MANIFEST = (
    PIPELINE_DIR / "manifiesto_division_eventos_nocturno_v1.csv"
)

SUMMARY_PATH = (
    PIPELINE_DIR / "resumen_pipeline_nocturno_v1.txt"
)

RANDOM_SEED = 20261114


# ============================================================
# UTILIDADES
# ============================================================

def parse_two_numbers(
    value: Any,
) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None

    if len(value) < 2:
        return None

    try:
        start = float(value[0])
        end = float(value[1])
    except (TypeError, ValueError):
        return None

    if not (
        math.isfinite(start)
        and math.isfinite(end)
        and start < end
    ):
        return None

    return [start, end]


def normalize_seizure_times(
    value: Any,
) -> list[float] | None:
    direct = parse_two_numbers(value)

    if direct is not None:
        return direct

    if not isinstance(value, str):
        return None

    cleaned = value.strip()

    if cleaned.lower() in {
        "",
        "none",
        "null",
        "nan",
        "[]",
    }:
        return None

    candidate: Any = cleaned

    for _ in range(3):
        parsed = None

        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            try:
                parsed = ast.literal_eval(candidate)
            except (ValueError, SyntaxError, TypeError):
                parsed = None

        if isinstance(parsed, str):
            candidate = parsed.strip()
            continue

        numeric = parse_two_numbers(parsed)

        if numeric is not None:
            return numeric

        break

    numbers = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        cleaned,
    )

    if len(numbers) >= 2:
        return parse_two_numbers(numbers[:2])

    return None


def normalize_event_id(
    value: Any,
) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        head = text[:-2]

        if head.replace("-", "").isdigit():
            text = head

    return text


def normalize_label(
    value: Any,
) -> int:
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


def read_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise RuntimeError(
            f"{path.name} no contiene una lista."
        )

    return [
        item
        for item in data
        if isinstance(item, dict)
    ]


def verify_flattened(
    path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        low_memory=False,
    )

    required = {
        "eventId",
        "userId",
        "type",
        "typeStr",
    }

    required.update(
        f"M{i:03d}"
        for i in range(125)
    )

    missing = required.difference(
        set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Faltan columnas en allData.csv:\n"
            + "\n".join(
                sorted(missing)[:30]
            )
        )

    magnitude_cols = [
        f"M{i:03d}"
        for i in range(125)
    ]

    incomplete = int(
        df[magnitude_cols]
        .isna()
        .any(axis=1)
        .sum()
    )

    if incomplete:
        raise RuntimeError(
            f"Hay {incomplete} filas incompletas."
        )

    output = df.copy()

    output["event_id_norm"] = (
        output["eventId"]
        .map(normalize_event_id)
    )

    output["_label"] = (
        output["type"]
        .map(normalize_label)
    )

    return output


def event_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    table = (
        df.groupby(
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

    return table


def split_ids(
    events: pd.DataFrame,
) -> tuple[
    set[str],
    set[str],
    set[str],
]:
    event_ids = events[
        "event_id_norm"
    ].to_numpy()

    labels = events[
        "label"
    ].to_numpy()

    train_ids, temp_ids = train_test_split(
        event_ids,
        test_size=0.40,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    temp = events.loc[
        events["event_id_norm"]
        .isin(temp_ids)
    ].copy()

    val_ids, test_ids = train_test_split(
        temp["event_id_norm"].to_numpy(),
        test_size=0.50,
        random_state=RANDOM_SEED + 1,
        stratify=temp["label"].to_numpy(),
    )

    return (
        set(train_ids),
        set(val_ids),
        set(test_ids),
    )


def write_split(
    df: pd.DataFrame,
    ids: set[str],
    path: Path,
) -> pd.DataFrame:
    subset = df.loc[
        df["event_id_norm"].isin(ids)
    ].copy()

    subset = subset.drop(
        columns=[
            "event_id_norm",
            "_label",
        ],
        errors="ignore",
    )

    subset.to_csv(
        path,
        index=False,
    )

    return subset


def metrics_for_split(
    path: Path,
) -> dict[str, Any]:
    df = verify_flattened(path)
    events = event_table(df)

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
        "user_set": set(
            df["userId"]
            .astype(str)
        ),
        "event_set": set(
            events["event_id_norm"]
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    PIPELINE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        POSITIVE_JSON,
        NEGATIVE_JSON,
        BASE_CONFIG_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró:\n{path}"
            )

    positives = read_json_list(
        POSITIVE_JSON
    )

    negatives = read_json_list(
        NEGATIVE_JSON
    )

    # Normalizar seizureTimes en positivos.
    invalid_seizure_times: list[str] = []

    for event in positives:
        event_id = normalize_event_id(
            event.get("id")
        )

        normalized = normalize_seizure_times(
            event.get("seizureTimes")
        )

        event["seizureTimes"] = normalized

        if normalized is None:
            invalid_seizure_times.append(
                event_id
            )

    if invalid_seizure_times:
        raise RuntimeError(
            "Hay positivos con seizureTimes inválido: "
            + ", ".join(
                invalid_seizure_times
            )
        )

    all_events = positives + negatives

    with ALL_JSON.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            all_events,
            handle,
            ensure_ascii=False,
            allow_nan=True,
            separators=(",", ":"),
        )

    # Cargar configuración existente y reutilizar únicamente
    # las opciones de procesamiento necesarias para flattenData.
    with BASE_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = json.load(handle)

    validate_datapoints = bool(
        config
        .get("dataProcessing", {})
        .get("validateDatapoints", False)
    )

    print("")
    print("=" * 78)
    print("1/3 APLANANDO DATASET NOCTURNO")
    print("=" * 78)

    if ALL_CSV.exists():
        ALL_CSV.unlink()

    flattenData.flattenOsdb(
        inFname=str(ALL_JSON),
        outFname=str(ALL_CSV),
        debug=False,
        validate_datapoints=validate_datapoints,
        config=config,
    )

    all_df = verify_flattened(
        ALL_CSV
    )

    events = event_table(
        all_df
    )

    print(
        f"Eventos recuperados tras aplanado: "
        f"{len(events)}/{len(all_events)}"
    )

    print(
        f"Positivos: "
        f"{int((events['label'] == 1).sum())}"
    )

    print(
        f"Negativos: "
        f"{int((events['label'] == 0).sum())}"
    )

    if events["label"].nunique() != 2:
        raise RuntimeError(
            "El aplanado no contiene las dos clases."
        )

    print("")
    print("=" * 78)
    print("2/3 DIVISIÓN POR EVENTO")
    print("=" * 78)

    train_ids, val_ids, test_ids = (
        split_ids(events)
    )

    write_split(
        all_df,
        train_ids,
        TRAIN_CSV,
    )

    write_split(
        all_df,
        val_ids,
        VAL_CSV,
    )

    write_split(
        all_df,
        test_ids,
        TEST_CSV,
    )

    print("")
    print("=" * 78)
    print("3/3 VERIFICACIÓN")
    print("=" * 78)

    train_m = metrics_for_split(
        TRAIN_CSV
    )

    val_m = metrics_for_split(
        VAL_CSV
    )

    test_m = metrics_for_split(
        TEST_CSV
    )

    if (
        train_m["event_set"]
        & val_m["event_set"]
        or
        train_m["event_set"]
        & test_m["event_set"]
        or
        val_m["event_set"]
        & test_m["event_set"]
    ):
        raise RuntimeError(
            "Se detectó fuga de eventId."
        )

    # Manifiesto por evento.
    manifest_rows = []

    for split_name, ids in [
        ("train", train_ids),
        ("validation", val_ids),
        ("test", test_ids),
    ]:
        subset = events.loc[
            events["event_id_norm"].isin(ids)
        ].copy()

        subset["split"] = split_name
        manifest_rows.append(subset)

    manifest = pd.concat(
        manifest_rows,
        ignore_index=True,
    )

    manifest.to_csv(
        SPLIT_MANIFEST,
        index=False,
        encoding="utf-8-sig",
    )

    train_val_users = (
        train_m["user_set"]
        & val_m["user_set"]
    )

    train_test_users = (
        train_m["user_set"]
        & test_m["user_set"]
    )

    val_test_users = (
        val_m["user_set"]
        & test_m["user_set"]
    )

    recovered_ids = set(
        events["event_id_norm"]
    )

    original_ids = {
        normalize_event_id(
            event.get("id")
        )
        for event in all_events
    }

    missing_after_flatten = (
        original_ids - recovered_ids
    )

    lines = [
        "PIPELINE NOCTURNO OSDB V1",
        "=" * 78,
        "",
        "ALCANCE",
        (
            "Detector limitado a dormido/descansando. "
            "Split por EVENTO, no por usuario."
        ),
        (
            "Solo hay 3 usuarios positivos, por lo que esta división "
            "es adecuada para prueba de concepto, no para estimar "
            "generalización a usuarios nuevos."
        ),
        "",
        "APLANADO",
        f"Eventos originales: {len(all_events)}",
        f"Eventos recuperados: {len(events)}",
        f"Eventos positivos recuperados: "
        f"{int((events['label'] == 1).sum())}",
        f"Eventos negativos recuperados: "
        f"{int((events['label'] == 0).sum())}",
        f"Filas/ventanas totales: {len(all_df)}",
        (
            "Eventos perdidos tras aplanado: "
            + (
                ", ".join(
                    sorted(missing_after_flatten)
                )
                if missing_after_flatten
                else "ninguno"
            )
        ),
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
        "FUGA DE eventId",
        "Train ∩ Validation: 0",
        "Train ∩ Test: 0",
        "Validation ∩ Test: 0",
        "",
        "SUPERPOSICIÓN DE userId (INFORMATIVA)",
        f"Train ∩ Validation: {len(train_val_users)}",
        f"Train ∩ Test: {len(train_test_users)}",
        f"Validation ∩ Test: {len(val_test_users)}",
        "",
        "VEREDICTO",
        "Fuga de eventId: NO",
        "TEST queda reservado.",
        (
            "No interpretar este split como validación independiente "
            "por paciente porque existe superposición de usuarios."
        ),
        "",
        "SIGUIENTE PASO",
        (
            "Entrenar únicamente con TRAIN y ajustar únicamente "
            "con VALIDATION. TEST permanecerá cerrado."
        ),
    ]

    SUMMARY_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

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

    print("Fuga de eventId: 0")
    print(f"Resumen: {SUMMARY_PATH}")
    print("")
    print(
        "Todavía NO se ha entrenado ningún modelo "
        "y TEST permanece reservado."
    )


if __name__ == "__main__":
    main()

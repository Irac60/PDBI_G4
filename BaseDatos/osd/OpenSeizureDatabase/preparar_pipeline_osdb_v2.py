"""
preparar_pipeline_osdb_v2.py
============================

Prepara el pipeline V2 sin entrenar:

1. Crea/valida nnConfig_prototipo_v2.json.
2. Selecciona los 240 eventos del subconjunto V2.
3. Normaliza seizureTimes heredados.
4. Aplana los eventos en filas de 5 s con M000..M124.
5. Divide por eventId en train/validation/test.
6. Comprueba que no exista fuga de eventId.
7. Genera resúmenes y manifiesto.

NO extrae features DeepEpi.
NO entrena.
NO usa test para ajustar nada.

Ejecución:
    python .\preparar_pipeline_osdb_v2.py
"""

from __future__ import annotations

import ast
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from user_tools.nnTraining2 import flattenData, selectData, splitData


# ============================================================
# RUTAS
# ============================================================

REPO_DIR = Path(r"C:\Users\cesar\osd\OpenSeizureDatabase")
CONFIG_PATH = REPO_DIR / "nnConfig_prototipo_v2.json"
SUBSET_DIR = Path(r"C:\Users\cesar\osd\osdb_subset_v2")
PIPELINE_DIR = Path(r"C:\Users\cesar\osd\pipeline_prototipo_v2")

# True: elimina los archivos intermedios V2 y los vuelve a generar.
REGENERAR = True


# ============================================================
# UTILIDADES
# ============================================================

def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró la configuración:\n{path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_two_finite_numbers(
    value: Any,
) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
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


def normalize_seizure_times_value(
    value: Any,
) -> tuple[list[float] | None, str]:
    direct = parse_two_finite_numbers(value)

    if direct is not None:
        return direct, "already_numeric"

    if value is None:
        return None, "missing"

    if not isinstance(value, str):
        return None, "invalid"

    cleaned = value.strip()

    if cleaned.lower() in {
        "",
        "none",
        "null",
        "nan",
        "[]",
    }:
        return None, "missing"

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

        numeric = parse_two_finite_numbers(parsed)

        if numeric is not None:
            return numeric, "parsed_string"

        break

    numbers = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        cleaned,
    )

    if len(numbers) >= 2:
        numeric = parse_two_finite_numbers(numbers[:2])

        if numeric is not None:
            return numeric, "parsed_regex"

    return None, "invalid"


def normalize_seizure_times_file(
    json_path: Path,
    report_path: Path,
) -> dict[str, int]:
    with json_path.open("r", encoding="utf-8") as handle:
        events = json.load(handle)

    if not isinstance(events, list):
        raise ValueError("allData.json no contiene una lista.")

    counters = {
        "events_total": len(events),
        "already_numeric": 0,
        "parsed_string": 0,
        "parsed_regex": 0,
        "missing": 0,
        "invalid": 0,
        "changed": 0,
    }

    invalid_ids: list[str] = []

    for event in events:
        if not isinstance(event, dict):
            counters["invalid"] += 1
            continue

        original = event.get("seizureTimes")
        normalized, status = normalize_seizure_times_value(
            original
        )

        counters[status] = counters.get(status, 0) + 1

        if normalized != original:
            event["seizureTimes"] = normalized
            counters["changed"] += 1

        if status == "invalid":
            invalid_ids.append(
                str(event.get("id", "<sin_id>"))
            )

    backup_path = json_path.with_name(
        "allData_antes_corregir_seizureTimes.json"
    )

    if not backup_path.exists():
        shutil.copy2(json_path, backup_path)

    temporary_path = json_path.with_suffix(".json.tmp")

    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            events,
            handle,
            ensure_ascii=False,
            allow_nan=True,
            separators=(",", ":"),
        )

    temporary_path.replace(json_path)

    lines = [
        "NORMALIZACIÓN seizureTimes — PIPELINE V2",
        "=" * 72,
        f"Archivo: {json_path}",
        f"Respaldo: {backup_path}",
        "",
    ]

    for key, value in counters.items():
        lines.append(f"{key}: {value}")

    if invalid_ids:
        lines.extend(
            [
                "",
                "IDs inválidos:",
                *invalid_ids,
            ]
        )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return counters


def normalize_binary_label(value: Any) -> int:
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


def verify_flattened_csv(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se generó:\n{path}")

    df = pd.read_csv(path, low_memory=False)

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

    missing = required.difference(df.columns)

    if missing:
        raise RuntimeError(
            "Faltan columnas en allData.csv:\n"
            + "\n".join(sorted(missing)[:30])
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
            f"Hay {incomplete} filas con M000..M124 incompletas."
        )

    df = df.copy()
    df["_label"] = df["type"].map(
        normalize_binary_label
    )

    return df


def unique_event_table(
    df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    grouped = (
        df.groupby("eventId", dropna=False)
        .agg(
            userId=("userId", "first"),
            typeStr=("typeStr", "first"),
            label=("_label", "first"),
            n_rows=("eventId", "size"),
        )
        .reset_index()
    )

    grouped.insert(0, "split", split_name)
    return grouped


def overlap_as_strings(
    first: pd.Series,
    second: pd.Series,
) -> set[str]:
    return set(first.astype(str)).intersection(
        set(second.astype(str))
    )


def split_metrics(
    df: pd.DataFrame,
    events: pd.DataFrame,
) -> dict[str, int]:
    return {
        "rows": len(df),
        "events": len(events),
        "positive_rows": int((df["_label"] == 1).sum()),
        "negative_rows": int((df["_label"] == 0).sum()),
        "positive_events": int((events["label"] == 1).sum()),
        "negative_events": int((events["label"] == 0).sum()),
        "users": int(df["userId"].astype(str).nunique()),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    config = load_config(CONFIG_PATH)

    expected_cache = str(
        config["osdbConfig"]["cacheDir"]
    ).replace("\\", "/")

    actual_subset = str(SUBSET_DIR).replace("\\", "/")

    if expected_cache.lower() != actual_subset.lower():
        raise RuntimeError(
            "cacheDir de la configuración no coincide con "
            f"SUBSET_DIR.\nConfig: {expected_cache}\n"
            f"Esperado: {actual_subset}"
        )

    for filename in config["osdbConfig"]["osdbFiles"]:
        path = SUBSET_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Falta el JSON V2:\n{path}"
            )

    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

    names = config["dataFileNames"]

    all_json = PIPELINE_DIR / names["allDataFileJson"]
    all_csv = PIPELINE_DIR / names["allDataFileCsv"]
    train_csv = PIPELINE_DIR / names["trainDataFileCsv"]
    val_csv = PIPELINE_DIR / names["valDataFileCsv"]
    test_csv = PIPELINE_DIR / names["testDataFileCsv"]

    correction_report = (
        PIPELINE_DIR
        / "resumen_correccion_seizureTimes_v2.txt"
    )

    pipeline_summary = (
        PIPELINE_DIR
        / "resumen_pipeline_v2.txt"
    )

    split_manifest_path = (
        PIPELINE_DIR
        / "manifiesto_division_eventos_v2.csv"
    )

    if REGENERAR:
        for path in [
            all_json,
            all_csv,
            train_csv,
            val_csv,
            test_csv,
            correction_report,
            pipeline_summary,
            split_manifest_path,
        ]:
            if path.exists():
                path.unlink()

    # --------------------------------------------------------
    # Selección
    # --------------------------------------------------------
    print("")
    print("=" * 78)
    print("1/4 SELECCIÓN DE EVENTOS V2")
    print("=" * 78)

    selectData.selectData(
        config,
        outDir=str(PIPELINE_DIR),
        debug=True,
    )

    if not all_json.exists():
        raise RuntimeError("selectData no creó allData.json.")

    # --------------------------------------------------------
    # Normalización de seizureTimes
    # --------------------------------------------------------
    print("")
    print("=" * 78)
    print("2/4 NORMALIZACIÓN DE seizureTimes")
    print("=" * 78)

    normalization_counts = normalize_seizure_times_file(
        all_json,
        correction_report,
    )

    print(normalization_counts)

    # --------------------------------------------------------
    # Aplanado
    # --------------------------------------------------------
    print("")
    print("=" * 78)
    print("3/4 APLANADO A BLOQUES DE 5 s")
    print("=" * 78)

    if all_csv.exists():
        all_csv.unlink()

    validate_datapoints = bool(
        config
        .get("dataProcessing", {})
        .get("validateDatapoints", False)
    )

    flattenData.flattenOsdb(
        inFname=str(all_json),
        outFname=str(all_csv),
        debug=False,
        validate_datapoints=validate_datapoints,
        config=config,
    )

    all_df = verify_flattened_csv(all_csv)

    total_events = int(
        all_df["eventId"]
        .astype(str)
        .nunique()
    )

    positive_events = int(
        all_df.loc[
            all_df["_label"] == 1,
            "eventId",
        ]
        .astype(str)
        .nunique()
    )

    negative_events = int(
        all_df.loc[
            all_df["_label"] == 0,
            "eventId",
        ]
        .astype(str)
        .nunique()
    )

    print(f"Filas: {len(all_df)}")
    print(f"Eventos: {total_events}")
    print(f"Eventos positivos: {positive_events}")
    print(f"Eventos negativos: {negative_events}")

    # --------------------------------------------------------
    # División por evento
    # --------------------------------------------------------
    print("")
    print("=" * 78)
    print("4/4 DIVISIÓN TRAIN / VALIDATION / TEST")
    print("=" * 78)

    splitData.splitData(
        configObj=config,
        kFold=1,
        nestedKfold=1,
        outDir=str(PIPELINE_DIR),
        debug=True,
    )

    splits: dict[str, pd.DataFrame] = {}

    for name, path in [
        ("train", train_csv),
        ("validation", val_csv),
        ("test", test_csv),
    ]:
        split_df = verify_flattened_csv(path)
        splits[name] = split_df

    event_tables = {
        name: unique_event_table(df, name)
        for name, df in splits.items()
    }

    overlap_train_val = overlap_as_strings(
        event_tables["train"]["eventId"],
        event_tables["validation"]["eventId"],
    )

    overlap_train_test = overlap_as_strings(
        event_tables["train"]["eventId"],
        event_tables["test"]["eventId"],
    )

    overlap_val_test = overlap_as_strings(
        event_tables["validation"]["eventId"],
        event_tables["test"]["eventId"],
    )

    if (
        overlap_train_val
        or overlap_train_test
        or overlap_val_test
    ):
        raise RuntimeError(
            "Se detectó fuga de eventId entre conjuntos."
        )

    manifest = pd.concat(
        [
            event_tables["train"],
            event_tables["validation"],
            event_tables["test"],
        ],
        ignore_index=True,
    )

    manifest.to_csv(
        split_manifest_path,
        index=False,
    )

    metrics = {
        name: split_metrics(
            splits[name],
            event_tables[name],
        )
        for name in [
            "train",
            "validation",
            "test",
        ]
    }

    users_train_val = overlap_as_strings(
        splits["train"]["userId"],
        splits["validation"]["userId"],
    )

    users_train_test = overlap_as_strings(
        splits["train"]["userId"],
        splits["test"]["userId"],
    )

    users_val_test = overlap_as_strings(
        splits["validation"]["userId"],
        splits["test"]["userId"],
    )

    lines = [
        "PIPELINE OSDB V2 PREPARADO",
        "=" * 78,
        "",
        f"Configuración: {CONFIG_PATH}",
        f"Subconjunto: {SUBSET_DIR}",
        f"Pipeline: {PIPELINE_DIR}",
        "",
        "APLANADO",
        f"Filas totales: {len(all_df)}",
        f"Eventos totales: {total_events}",
        f"Eventos positivos: {positive_events}",
        f"Eventos negativos: {negative_events}",
        f"Filas positivas: {int((all_df['_label'] == 1).sum())}",
        f"Filas negativas: {int((all_df['_label'] == 0).sum())}",
        "",
        "DIVISIÓN",
    ]

    for name in [
        "train",
        "validation",
        "test",
    ]:
        item = metrics[name]

        lines.extend(
            [
                "",
                name.upper(),
                f"Filas: {item['rows']}",
                f"Eventos: {item['events']}",
                f"Eventos positivos: {item['positive_events']}",
                f"Eventos negativos: {item['negative_events']}",
                f"Filas positivas: {item['positive_rows']}",
                f"Filas negativas: {item['negative_rows']}",
                f"Usuarios: {item['users']}",
            ]
        )

    lines.extend(
        [
            "",
            "FUGA DE eventId",
            f"Train ∩ Validation: {len(overlap_train_val)}",
            f"Train ∩ Test: {len(overlap_train_test)}",
            f"Validation ∩ Test: {len(overlap_val_test)}",
            "",
            "SUPERPOSICIÓN DE userId (informativa)",
            f"Train ∩ Validation: {len(users_train_val)}",
            f"Train ∩ Test: {len(users_train_test)}",
            f"Validation ∩ Test: {len(users_val_test)}",
            "",
            "VEREDICTO",
            "Fuga de eventId: NO",
            f"Eventos recuperados: {len(manifest)}/{total_events}",
            "TEST permanece reservado.",
        ]
    )

    pipeline_summary.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("")
    print("=" * 78)
    print("PIPELINE V2 PREPARADO CORRECTAMENTE")
    print("=" * 78)

    for name, item in metrics.items():
        print(
            f"{name}: {item['events']} eventos, "
            f"{item['rows']} filas, "
            f"{item['positive_events']} positivos, "
            f"{item['negative_events']} negativos"
        )

    print("Fuga de eventId: 0")
    print(f"Resumen: {pipeline_summary}")
    print(f"Manifiesto: {split_manifest_path}")
    print("")
    print("Todavía NO se ha entrenado ni evaluado TEST.")


if __name__ == "__main__":
    main()

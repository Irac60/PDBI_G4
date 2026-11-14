"""
aplanar_osdb_prototipo_V2.py
=========================

Tramo de preparación OSDB:
1. Carga nnConfig_prototipo_pytorch.json.
2. Si allData.json no existe, ejecuta selectData sobre el subconjunto.
3. Convierte allData.json a allData.csv usando flattenData COMO MÓDULO,
   de modo que sí se apliquen:
      - useSeizureTimesConstraint
      - seizureTimeMarginSeconds
      - validateDatapoints
      - dcNormalisation
4. Genera un resumen sencillo del CSV.

No divide train/validation/test.
No aumenta datos.
No entrena ningún modelo.

Ejecución desde la raíz del repositorio:
    python .\aplanar_osdb_prototipo.py
"""

from __future__ import annotations

import ast
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

from user_tools.nnTraining2 import flattenData, selectData


# ============================================================
# RUTAS
# ============================================================

REPO_DIR = Path(r"C:\Users\cesar\osd\OpenSeizureDatabase")
CONFIG_PATH = REPO_DIR / "nnConfig_prototipo_pytorch.json"
PIPELINE_DIR = Path(r"C:\Users\cesar\osd\pipeline_prototipo")

# False: conserva archivos ya creados.
# True: vuelve a generar allData.json y allData.csv.
FORZAR_REGENERACION = False


# ============================================================
# FUNCIONES
# ============================================================

def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró la configuración:\n{path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)



def _parse_two_finite_numbers(value) -> list[float] | None:
    """
    Convierte value en [inicio, fin] si contiene dos números finitos.
    No invierte ni inventa límites; un valor inválido devuelve None.
    """
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None

    try:
        start = float(value[0])
        end = float(value[1])
    except (TypeError, ValueError):
        return None

    if not (math.isfinite(start) and math.isfinite(end)):
        return None

    if start > end:
        return None

    return [start, end]


def normalize_seizure_times_value(value):
    """
    Normaliza seizureTimes.

    Casos admitidos:
      [-38.0, 76.0]
      "[-38.0, 76.0]"
      "'[-38.0, 76.0]'"
      "[-38, 76]"

    Devuelve:
      (valor_normalizado, estado)
    """
    direct = _parse_two_finite_numbers(value)

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

    candidate = cleaned

    # Algunas versiones pueden contener una cadena JSON dentro de otra.
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

        numeric = _parse_two_finite_numbers(parsed)

        if numeric is not None:
            return numeric, "parsed_string"

        break

    # Respaldo para formatos heredados con corchetes/comas extraños.
    numbers = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        cleaned,
    )

    if len(numbers) >= 2:
        numeric = _parse_two_finite_numbers(numbers[:2])

        if numeric is not None:
            return numeric, "parsed_regex"

    return None, "invalid"


def normalize_seizure_times_file(
    json_path: Path,
    report_path: Path,
) -> dict[str, int]:
    """
    Corrige seizureTimes en allData.json y crea una copia de respaldo.
    El archivo contiene solo el subconjunto reducido, por lo que puede
    cargarse de forma segura en memoria.
    """
    if not json_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo:\n{json_path}"
        )

    with json_path.open("r", encoding="utf-8") as handle:
        events = json.load(handle)

    if not isinstance(events, list):
        raise ValueError(
            "allData.json no contiene una lista de eventos."
        )

    counters = {
        "events_total": len(events),
        "already_numeric": 0,
        "parsed_string": 0,
        "parsed_regex": 0,
        "missing": 0,
        "invalid": 0,
        "changed": 0,
    }

    invalid_event_ids: list[str] = []

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
            invalid_event_ids.append(
                str(event.get("id", "<sin_id>"))
            )

    backup_path = json_path.with_name(
        json_path.stem
        + "_antes_corregir_seizureTimes.json"
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
        "NORMALIZACIÓN seizureTimes",
        "=" * 72,
        f"Archivo: {json_path}",
        f"Respaldo: {backup_path}",
        "",
    ]

    for key, value in counters.items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append(
        "Los valores missing/invalid quedan como null; "
        "flattenData aplicará su intervalo predeterminado."
    )

    if invalid_event_ids:
        lines.append("")
        lines.append("IDs con formato inválido:")
        lines.extend(invalid_event_ids)

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return counters


def verify_expected_columns(df: pd.DataFrame) -> None:
    required_metadata = [
        "eventId",
        "userId",
        "typeStr",
        "type",
        "dataTime",
        "osdAlarmState",
        "osdSpecPower",
        "osdRoiPower",
        "hr",
        "o2sat",
    ]

    required_magnitude = [
        f"M{i:03d}"
        for i in range(125)
    ]

    missing = [
        column
        for column in required_metadata + required_magnitude
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Faltan columnas esperadas en allData.csv:\n"
            + "\n".join(missing[:30])
        )


def count_complete_magnitude_rows(df: pd.DataFrame) -> int:
    magnitude_columns = [
        f"M{i:03d}"
        for i in range(125)
    ]

    return int(
        df[magnitude_columns]
        .notna()
        .all(axis=1)
        .sum()
    )


def write_summary(
    df: pd.DataFrame,
    summary_path: Path,
    all_json_path: Path,
    all_csv_path: Path,
    config: dict,
) -> None:
    type_counts = Counter(
        df["type"].astype(int).tolist()
    )

    type_str_counts = (
        df["typeStr"]
        .fillna("<VACÍO>")
        .astype(str)
        .value_counts()
    )

    unique_events_total = int(
        df["eventId"].astype(str).nunique()
    )

    unique_positive_events = int(
        df.loc[df["type"] == 1, "eventId"]
        .astype(str)
        .nunique()
    )

    unique_negative_events = int(
        df.loc[df["type"] == 0, "eventId"]
        .astype(str)
        .nunique()
    )

    positive_rows = int((df["type"] == 1).sum())
    negative_rows = int((df["type"] == 0).sum())
    other_rows = int(
        (~df["type"].isin([0, 1])).sum()
    )

    complete_magnitude_rows = (
        count_complete_magnitude_rows(df)
    )

    magnitude_columns = [
        f"M{i:03d}"
        for i in range(125)
    ]

    all_magnitude_values = (
        df[magnitude_columns]
        .apply(pd.to_numeric, errors="coerce")
    )

    magnitude_min = float(
        all_magnitude_values.min().min()
    )
    magnitude_max = float(
        all_magnitude_values.max().max()
    )
    magnitude_mean = float(
        all_magnitude_values.stack().mean()
    )

    lines = [
        "RESUMEN DEL APLANADO OSDB — PROTOTIPO",
        "=" * 72,
        "",
        f"Configuración: {CONFIG_PATH}",
        f"Entrada JSON: {all_json_path}",
        f"Salida CSV: {all_csv_path}",
        "",
        "CONFIGURACIÓN APLICADA",
        f"useSeizureTimesConstraint: "
        f"{config.get('useSeizureTimesConstraint')}",
        f"seizureTimeMarginSeconds: "
        f"{config.get('seizureTimeMarginSeconds')}",
        f"validateDatapoints: "
        f"{config.get('dataProcessing', {}).get('validateDatapoints')}",
        f"dcNormalisation: "
        f"{config.get('dataProcessing', {}).get('dcNormalisation')}",
        "",
        "DIMENSIONES",
        f"Filas/datapoints: {len(df)}",
        f"Columnas: {len(df.columns)}",
        f"Eventos únicos totales: {unique_events_total}",
        "",
        "EVENTOS ÚNICOS",
        f"Positivos (Seizure): {unique_positive_events}",
        f"Negativos (False Alarm + nda): "
        f"{unique_negative_events}",
        "",
        "FILAS DE CINCO SEGUNDOS",
        f"Positivas type=1: {positive_rows}",
        f"Negativas type=0: {negative_rows}",
        f"Otras: {other_rows}",
        f"Filas con M000..M124 completas: "
        f"{complete_magnitude_rows}",
        "",
        "DISTRIBUCIÓN type",
        f"{dict(type_counts)}",
        "",
        "DISTRIBUCIÓN typeStr",
    ]

    for label, count in type_str_counts.items():
        lines.append(f"{label}: {int(count)}")

    lines.extend(
        [
            "",
            "RANGO DE MAGNITUD M000..M124",
            f"Mínimo: {magnitude_min:.6f}",
            f"Máximo: {magnitude_max:.6f}",
            f"Media global: {magnitude_mean:.6f}",
            "",
            "INTERPRETACIÓN",
            "Cada fila representa un datapoint de 5 s.",
            "M000..M124 son las 125 muestras de magnitud.",
            "type=1 corresponde a Seizure.",
            "type=0 corresponde a False Alarm o nda.",
        ]
    )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:
    PIPELINE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = load_config(CONFIG_PATH)

    all_json_name = config[
        "dataFileNames"
    ].get(
        "allDataFileJson",
        "allData.json",
    )

    all_csv_name = config[
        "dataFileNames"
    ].get(
        "allDataFileCsv",
        "allData.csv",
    )

    all_json_path = (
        PIPELINE_DIR / all_json_name
    )

    all_csv_path = (
        PIPELINE_DIR / all_csv_name
    )

    summary_path = (
        PIPELINE_DIR /
        "resumen_aplanado_prototipo.txt"
    )

    normalization_report_path = (
        PIPELINE_DIR /
        "resumen_correccion_seizureTimes.txt"
    )

    if FORZAR_REGENERACION:
        if all_json_path.exists():
            all_json_path.unlink()

        if all_csv_path.exists():
            all_csv_path.unlink()

    # --------------------------------------------------------
    # Selección
    # --------------------------------------------------------
    if not all_json_path.exists():
        print("")
        print("allData.json no existe.")
        print("Ejecutando selectData...")

        selectData.selectData(
            config,
            outDir=str(PIPELINE_DIR),
            debug=True,
        )
    else:
        print("")
        print(
            "allData.json ya existe; "
            "no se volverá a seleccionar."
        )

    if not all_json_path.exists():
        raise RuntimeError(
            "selectData terminó, pero no se creó "
            f"{all_json_path}"
        )

    # --------------------------------------------------------
    # Normalizar seizureTimes heredados
    # --------------------------------------------------------
    print("")
    print("Normalizando seizureTimes en allData.json...")

    normalization_counts = normalize_seizure_times_file(
        json_path=all_json_path,
        report_path=normalization_report_path,
    )

    print(
        "Eventos corregidos: "
        f"{normalization_counts['changed']}"
    )
    print(
        "Cadenas convertidas a listas: "
        f"{normalization_counts['parsed_string'] + normalization_counts['parsed_regex']}"
    )
    print(
        "Valores inválidos sustituidos por null: "
        f"{normalization_counts['invalid']}"
    )
    print(
        "Informe de corrección: "
        f"{normalization_report_path}"
    )

    # Una ejecución fallida de flattenData puede haber dejado un CSV
    # parcial. Se elimina siempre para generar uno completo desde cero.
    if all_csv_path.exists():
        print("")
        print(
            "Eliminando allData.csv anterior o parcial "
            "para regenerarlo..."
        )
        all_csv_path.unlink()

    # --------------------------------------------------------
    # Aplanado con configuración
    # --------------------------------------------------------
    if not all_csv_path.exists():
        print("")
        print("Aplanando allData.json a allData.csv...")

        validate_datapoints = bool(
            config
            .get("dataProcessing", {})
            .get("validateDatapoints", False)
        )

        flattenData.flattenOsdb(
            inFname=str(all_json_path),
            outFname=str(all_csv_path),
            debug=False,
            validate_datapoints=validate_datapoints,
            config=config,
        )
    else:
        print("")
        print(
            "allData.csv ya existe; "
            "no se volverá a aplanar."
        )

    if not all_csv_path.exists():
        raise RuntimeError(
            "No se creó allData.csv."
        )

    # --------------------------------------------------------
    # Verificación
    # --------------------------------------------------------
    print("")
    print("Leyendo allData.csv para verificarlo...")

    df = pd.read_csv(
        all_csv_path,
        low_memory=False,
    )

    verify_expected_columns(df)

    write_summary(
        df=df,
        summary_path=summary_path,
        all_json_path=all_json_path,
        all_csv_path=all_csv_path,
        config=config,
    )

    print("")
    print("=" * 72)
    print("APLANADO COMPLETADO")
    print("=" * 72)
    print(f"Filas: {len(df)}")
    print(f"Columnas: {len(df.columns)}")
    print(
        "Eventos únicos: "
        f"{df['eventId'].astype(str).nunique()}"
    )
    print(
        "Filas positivas: "
        f"{int((df['type'] == 1).sum())}"
    )
    print(
        "Filas negativas: "
        f"{int((df['type'] == 0).sum())}"
    )
    print(f"CSV: {all_csv_path}")
    print(f"Resumen: {summary_path}")
    print(
        "Corrección seizureTimes: "
        f"{normalization_report_path}"
    )


if __name__ == "__main__":
    main()

"""
inspeccionar_osdb.py
--------------------
Inspecciona la liberación OSDB V1.11 sin cargar los JSON completos.

Genera:
  - inventario_osdb.txt
  - conteos_subtipos_osdb.csv
  - conteos_fuentes_osdb.csv

Requisitos:
  pandas (ya instalado en el entorno del proyecto)

Uso desde la raíz de OpenSeizureDatabase:
  python inspeccionar_osdb.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATA_DIR = Path(r"C:\Users\cesar\osd\osdb_full_v1.11")
OUTPUT_DIR = Path(r"C:\Users\cesar\osd\inventario_osdb")

EXPECTED_JSON = [
    "osdb_3min_allSeizures.json",
    "osdb_3min_falseAlarms.json",
    "osdb_3min_ndaEvents.json",
    "osdb_3min_fallEvents.json",
]


# ============================================================
# UTILIDADES
# ============================================================

def human_size(size_bytes: int) -> str:
    """Convierte bytes a una representación legible."""
    value = float(size_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]

    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0

    return f"{size_bytes} B"


def read_first_json_array_object(
    file_path: Path,
    chunk_size: int = 1024 * 1024,
    max_bytes: int = 100 * 1024 * 1024,
) -> dict[str, Any]:
    """
    Lee únicamente el primer objeto de un archivo JSON cuya raíz es una lista.
    No carga el archivo completo.
    """
    decoder = json.JSONDecoder()
    buffer = ""
    bytes_read = 0
    array_started = False

    with file_path.open("r", encoding="utf-8-sig") as handle:
        while bytes_read < max_bytes:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            bytes_read += len(chunk.encode("utf-8", errors="ignore"))
            buffer += chunk

            if not array_started:
                stripped = buffer.lstrip()

                if not stripped:
                    continue

                if not stripped.startswith("["):
                    raise ValueError(
                        f"La raíz de {file_path.name} no parece ser una lista JSON."
                    )

                # Eliminar solo el primer '[' y espacios posteriores.
                start = buffer.find("[")
                buffer = buffer[start + 1 :].lstrip()
                array_started = True

            if not buffer:
                continue

            try:
                obj, _ = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                continue

            if not isinstance(obj, dict):
                raise ValueError(
                    f"El primer elemento de {file_path.name} no es un objeto."
                )

            return obj

    raise RuntimeError(
        f"No se pudo leer el primer evento de {file_path.name} "
        f"tras {human_size(bytes_read)}."
    )


def find_column(
    columns: list[str],
    candidates: list[str],
) -> str | None:
    """Busca una columna ignorando mayúsculas y espacios."""
    normalized = {
        str(column).strip().lower(): str(column)
        for column in columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    return None


def summarize_csv(
    csv_path: Path,
    report_lines: list[str],
    subtype_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> None:
    """Resume un CSV de eventos sin modificarlo."""
    report_lines.append("")
    report_lines.append("=" * 78)
    report_lines.append(f"CSV: {csv_path.name}")
    report_lines.append("=" * 78)

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:
        report_lines.append(f"ERROR al leer CSV: {exc}")
        return

    report_lines.append(f"Filas: {len(df)}")
    report_lines.append(f"Columnas ({len(df.columns)}): {list(df.columns)}")

    type_col = find_column(
        list(df.columns),
        ["type", "eventType"],
    )
    subtype_col = find_column(
        list(df.columns),
        ["subType", "subtype", "eventSubType"],
    )
    user_col = find_column(
        list(df.columns),
        ["userId", "userid", "user_id"],
    )
    event_col = find_column(
        list(df.columns),
        ["eventId", "eventid", "id"],
    )
    source_col = find_column(
        list(df.columns),
        ["dataSource", "dataSourceName", "datasource"],
    )

    if event_col:
        report_lines.append(
            f"Eventos únicos según {event_col}: "
            f"{df[event_col].nunique(dropna=True)}"
        )

    if user_col:
        report_lines.append(
            f"Usuarios únicos según {user_col}: "
            f"{df[user_col].nunique(dropna=True)}"
        )

    if type_col:
        report_lines.append("")
        report_lines.append(f"Conteo por {type_col}:")

        counts = (
            df[type_col]
            .fillna("<VACÍO>")
            .astype(str)
            .value_counts(dropna=False)
        )

        for value, count in counts.items():
            report_lines.append(f"  {value}: {count}")

    if subtype_col:
        report_lines.append("")
        report_lines.append(f"Conteo por {subtype_col}:")

        counts = (
            df[subtype_col]
            .fillna("<VACÍO>")
            .astype(str)
            .value_counts(dropna=False)
        )

        for value, count in counts.items():
            report_lines.append(f"  {value}: {count}")

            subtype_rows.append(
                {
                    "source_file": csv_path.name,
                    "subtype": value,
                    "count": int(count),
                }
            )

    if source_col:
        report_lines.append("")
        report_lines.append(f"Conteo por {source_col}:")

        counts = (
            df[source_col]
            .fillna("<VACÍO>")
            .astype(str)
            .value_counts(dropna=False)
        )

        for value, count in counts.items():
            report_lines.append(f"  {value}: {count}")

            source_rows.append(
                {
                    "source_file": csv_path.name,
                    "data_source": value,
                    "count": int(count),
                }
            )


def summarize_first_json_event(
    json_path: Path,
    report_lines: list[str],
) -> None:
    """Inspecciona solo el primer evento de un JSON."""
    report_lines.append("")
    report_lines.append("=" * 78)
    report_lines.append(f"JSON: {json_path.name}")
    report_lines.append("=" * 78)

    try:
        event = read_first_json_array_object(json_path)
    except Exception as exc:
        report_lines.append(f"ERROR al inspeccionar el primer evento: {exc}")
        return

    report_lines.append(f"Claves del evento: {sorted(event.keys())}")
    report_lines.append(f"id: {event.get('id')}")
    report_lines.append(f"userId: {event.get('userId')}")
    report_lines.append(f"type: {event.get('type')}")
    report_lines.append(f"subType: {event.get('subType')}")
    report_lines.append(f"sampleFreq: {event.get('sampleFreq')}")
    report_lines.append(f"seizureTimes: {event.get('seizureTimes')}")
    report_lines.append(f"dataSourceName: {event.get('dataSourceName')}")

    datapoints = event.get("datapoints")

    if not isinstance(datapoints, list):
        report_lines.append("datapoints: ausente o no es una lista")
        return

    report_lines.append(f"Número de datapoints en el primer evento: {len(datapoints)}")

    if not datapoints:
        return

    first_dp = datapoints[0]

    if not isinstance(first_dp, dict):
        report_lines.append("El primer datapoint no es un objeto.")
        return

    report_lines.append(
        f"Claves del primer datapoint: {sorted(first_dp.keys())}"
    )

    raw_data = first_dp.get("rawData")
    raw_data_3d = first_dp.get("rawData3D")

    report_lines.append(
        "Longitud rawData: "
        + (
            str(len(raw_data))
            if isinstance(raw_data, list)
            else "ausente/no lista"
        )
    )

    report_lines.append(
        "Longitud rawData3D: "
        + (
            str(len(raw_data_3d))
            if isinstance(raw_data_3d, list)
            else "ausente/no lista"
        )
    )

    report_lines.append(f"hr: {first_dp.get('hr')}")
    report_lines.append(f"o2Sat: {first_dp.get('o2Sat')}")
    report_lines.append(f"alarmState: {first_dp.get('alarmState')}")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_lines: list[str] = []
    subtype_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    report_lines.append("INVENTARIO OPEN SEIZURE DATABASE")
    report_lines.append("=" * 78)
    report_lines.append(f"Carpeta de datos: {DATA_DIR}")
    report_lines.append("")

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"No existe la carpeta: {DATA_DIR}"
        )

    files = sorted(
        path
        for path in DATA_DIR.iterdir()
        if path.is_file()
    )

    report_lines.append("ARCHIVOS ENCONTRADOS")
    report_lines.append("-" * 78)

    for file_path in files:
        report_lines.append(
            f"{file_path.name}: {human_size(file_path.stat().st_size)}"
        )

    report_lines.append("")
    report_lines.append("COMPROBACIÓN DE JSON ESPERADOS")
    report_lines.append("-" * 78)

    for file_name in EXPECTED_JSON:
        path = DATA_DIR / file_name
        report_lines.append(
            f"{file_name}: {'OK' if path.exists() else 'FALTA'}"
        )

    # Resumir todos los CSV principales.
    for csv_path in sorted(DATA_DIR.glob("osdb_3min_*.csv")):
        summarize_csv(
            csv_path,
            report_lines,
            subtype_rows,
            source_rows,
        )

    # Inspeccionar solo el primer evento de cada JSON.
    for file_name in EXPECTED_JSON:
        json_path = DATA_DIR / file_name

        if json_path.exists():
            summarize_first_json_event(
                json_path,
                report_lines,
            )

    report_path = OUTPUT_DIR / "inventario_osdb.txt"
    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    subtype_path = OUTPUT_DIR / "conteos_subtipos_osdb.csv"

    with subtype_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_file", "subtype", "count"],
        )
        writer.writeheader()
        writer.writerows(subtype_rows)

    source_path = OUTPUT_DIR / "conteos_fuentes_osdb.csv"

    with source_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_file", "data_source", "count"],
        )
        writer.writeheader()
        writer.writerows(source_rows)

    print("")
    print("Inventario terminado.")
    print(f"Informe: {report_path}")
    print(f"Subtipos: {subtype_path}")
    print(f"Fuentes: {source_path}")


if __name__ == "__main__":
    main()

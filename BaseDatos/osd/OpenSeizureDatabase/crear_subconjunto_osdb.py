"""
crear_subconjunto_osdb.py
=========================

Crea un subconjunto pequeño de OSDB V1.11 SIN cargar los JSON completos
en memoria. Usa un muestreo reservoir reproducible.

Subconjunto inicial:
- 40 eventos Seizure / Tonic-Clonic
- 20 eventos False Alarm
- 20 eventos nda

Así se obtiene inicialmente:
- 40 eventos positivos
- 40 eventos negativos

Los archivos de salida conservan los nombres esperados por nnTraining2:
- osdb_3min_allSeizures.json
- osdb_3min_falseAlarms.json
- osdb_3min_ndaEvents.json

También genera:
- manifiesto_subconjunto.csv
- resumen_subconjunto.txt

Ejecución:
    python .\crear_subconjunto_osdb.py
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator


# ============================================================
# RUTAS Y TAMAÑOS DEL SUBCONJUNTO
# ============================================================

DATA_DIR = Path(r"C:\Users\cesar\osd\osdb_full_v1.11")
OUTPUT_DIR = Path(r"C:\Users\cesar\osd\osdb_subset_v1")

RANDOM_SEED = 42

N_TONIC_CLONIC = 40
N_FALSE_ALARMS = 20
N_NDA_EVENTS = 20

CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB


# ============================================================
# LECTOR INCREMENTAL DE UNA LISTA JSON
# Usa json.JSONDecoder, que tolera los NaN presentes en OSDB.
# ============================================================

def iter_json_array(
    file_path: Path,
    chunk_size: int = CHUNK_SIZE,
) -> Iterator[dict[str, Any]]:
    """
    Recorre, uno por uno, los objetos de una lista JSON almacenada
    en un archivo grande. No carga el archivo completo.
    """
    decoder = json.JSONDecoder()

    with file_path.open(
        "r",
        encoding="utf-8-sig",
        errors="strict",
    ) as handle:
        buffer = ""
        position = 0
        array_started = False
        finished = False

        while not finished:
            # Añadir datos si el buffer restante es pequeño.
            if (
                position >= len(buffer)
                or len(buffer) - position < 1024
            ):
                if position > 0:
                    buffer = buffer[position:]
                    position = 0

                chunk = handle.read(chunk_size)

                if chunk:
                    buffer += chunk
                elif not buffer.strip():
                    break

            # Saltar espacios.
            while (
                position < len(buffer)
                and buffer[position].isspace()
            ):
                position += 1

            if not array_started:
                if position >= len(buffer):
                    continue

                if buffer[position] != "[":
                    raise ValueError(
                        f"{file_path.name} no comienza con una lista JSON."
                    )

                position += 1
                array_started = True
                continue

            # Saltar espacios y comas entre objetos.
            while position < len(buffer):
                char = buffer[position]

                if char.isspace() or char == ",":
                    position += 1
                    continue

                break

            if position >= len(buffer):
                # Necesitamos otro bloque.
                continue

            if buffer[position] == "]":
                finished = True
                break

            try:
                obj, end_position = decoder.raw_decode(
                    buffer,
                    position,
                )
            except json.JSONDecodeError:
                # Probablemente el objeto está partido entre dos bloques.
                remaining = buffer[position:]

                chunk = handle.read(chunk_size)

                if not chunk:
                    raise

                buffer = remaining + chunk
                position = 0
                continue

            if not isinstance(obj, dict):
                raise ValueError(
                    f"Se esperaba un objeto en {file_path.name}, "
                    f"pero se encontró {type(obj).__name__}."
                )

            yield obj
            position = end_position


# ============================================================
# VALIDACIÓN BÁSICA DE UN EVENTO
# ============================================================

def valid_raw_datapoint_count(event: dict[str, Any]) -> int:
    """
    Cuenta datapoints con rawData de 125 valores.
    Es la entrada 1D compatible con el primer modelo.
    """
    datapoints = event.get("datapoints")

    if not isinstance(datapoints, list):
        return 0

    count = 0

    for datapoint in datapoints:
        if not isinstance(datapoint, dict):
            continue

        raw_data = datapoint.get("rawData")

        if isinstance(raw_data, list) and len(raw_data) >= 125:
            count += 1

    return count


def is_tonic_clonic(event: dict[str, Any]) -> bool:
    return (
        str(event.get("type", "")).strip().lower()
        == "seizure"
        and
        str(event.get("subType", "")).strip().lower()
        == "tonic-clonic"
        and
        valid_raw_datapoint_count(event) > 0
    )


def is_false_alarm(event: dict[str, Any]) -> bool:
    return (
        str(event.get("type", "")).strip().lower()
        == "false alarm"
        and
        valid_raw_datapoint_count(event) > 0
    )


def is_nda(event: dict[str, Any]) -> bool:
    return (
        str(event.get("type", "")).strip().lower()
        == "nda"
        and
        valid_raw_datapoint_count(event) > 0
    )


# ============================================================
# RESERVOIR SAMPLING
# ============================================================

@dataclass
class SampleResult:
    selected: list[dict[str, Any]]
    matching_events_seen: int
    total_events_seen: int


def reservoir_sample_events(
    file_path: Path,
    predicate: Callable[[dict[str, Any]], bool],
    sample_size: int,
    seed: int,
    category_name: str,
) -> SampleResult:
    """
    Selecciona una muestra aleatoria uniforme de los eventos que
    cumplen el filtro, recorriendo el archivo una sola vez.
    """
    rng = random.Random(seed)

    reservoir: list[dict[str, Any]] = []
    matching = 0
    total = 0

    print("")
    print(f"Procesando {category_name}: {file_path.name}")
    print(f"Tamaño solicitado: {sample_size} eventos")

    for event in iter_json_array(file_path):
        total += 1

        if total % 500 == 0:
            print(
                f"  Eventos leídos: {total}; "
                f"coincidentes: {matching}",
                flush=True,
            )

        if not predicate(event):
            continue

        matching += 1

        if len(reservoir) < sample_size:
            reservoir.append(event)
        else:
            replacement_index = rng.randrange(matching)

            if replacement_index < sample_size:
                reservoir[replacement_index] = event

    if matching < sample_size:
        raise RuntimeError(
            f"{category_name}: solo se encontraron {matching} "
            f"eventos válidos, menos de los {sample_size} solicitados."
        )

    print(
        f"  Final: {total} eventos leídos; "
        f"{matching} coincidentes; "
        f"{len(reservoir)} seleccionados."
    )

    return SampleResult(
        selected=reservoir,
        matching_events_seen=matching,
        total_events_seen=total,
    )


# ============================================================
# GUARDADO
# ============================================================

def save_json_events(
    events: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Guarda una lista de eventos sin indentación para reducir tamaño.
    allow_nan=True conserva los NaN presentes en la base original.
    """
    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            events,
            handle,
            ensure_ascii=False,
            allow_nan=True,
            separators=(",", ":"),
        )


def create_manifest_rows(
    source_name: str,
    selected_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for event in selected_events:
        datapoints = event.get("datapoints")
        total_datapoints = (
            len(datapoints)
            if isinstance(datapoints, list)
            else 0
        )

        rows.append(
            {
                "source_file": source_name,
                "event_id": event.get("id"),
                "user_id": event.get("userId"),
                "type": event.get("type"),
                "subtype": event.get("subType"),
                "data_source": event.get("dataSourceName"),
                "total_datapoints": total_datapoints,
                "valid_rawdata_125": valid_raw_datapoint_count(
                    event
                ),
                "seizure_times": repr(
                    event.get("seizureTimes")
                ),
            }
        )

    return rows


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seizures_path = (
        DATA_DIR / "osdb_3min_allSeizures.json"
    )
    false_alarms_path = (
        DATA_DIR / "osdb_3min_falseAlarms.json"
    )
    nda_path = (
        DATA_DIR / "osdb_3min_ndaEvents.json"
    )

    required_files = [
        seizures_path,
        false_alarms_path,
        nda_path,
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Faltan archivos:\n" + "\n".join(missing)
        )

    tonic_clonic_result = reservoir_sample_events(
        file_path=seizures_path,
        predicate=is_tonic_clonic,
        sample_size=N_TONIC_CLONIC,
        seed=RANDOM_SEED + 1,
        category_name="Tonic-Clonic",
    )

    false_alarm_result = reservoir_sample_events(
        file_path=false_alarms_path,
        predicate=is_false_alarm,
        sample_size=N_FALSE_ALARMS,
        seed=RANDOM_SEED + 2,
        category_name="False Alarm",
    )

    nda_result = reservoir_sample_events(
        file_path=nda_path,
        predicate=is_nda,
        sample_size=N_NDA_EVENTS,
        seed=RANDOM_SEED + 3,
        category_name="NDA",
    )

    output_seizures = (
        OUTPUT_DIR / "osdb_3min_allSeizures.json"
    )
    output_false_alarms = (
        OUTPUT_DIR / "osdb_3min_falseAlarms.json"
    )
    output_nda = (
        OUTPUT_DIR / "osdb_3min_ndaEvents.json"
    )

    save_json_events(
        tonic_clonic_result.selected,
        output_seizures,
    )
    save_json_events(
        false_alarm_result.selected,
        output_false_alarms,
    )
    save_json_events(
        nda_result.selected,
        output_nda,
    )

    manifest_rows: list[dict[str, Any]] = []

    manifest_rows.extend(
        create_manifest_rows(
            output_seizures.name,
            tonic_clonic_result.selected,
        )
    )
    manifest_rows.extend(
        create_manifest_rows(
            output_false_alarms.name,
            false_alarm_result.selected,
        )
    )
    manifest_rows.extend(
        create_manifest_rows(
            output_nda.name,
            nda_result.selected,
        )
    )

    manifest_path = (
        OUTPUT_DIR / "manifiesto_subconjunto.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_file",
                "event_id",
                "user_id",
                "type",
                "subtype",
                "data_source",
                "total_datapoints",
                "valid_rawdata_125",
                "seizure_times",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    positive_windows = sum(
        valid_raw_datapoint_count(event)
        for event in tonic_clonic_result.selected
    )

    negative_windows_false = sum(
        valid_raw_datapoint_count(event)
        for event in false_alarm_result.selected
    )

    negative_windows_nda = sum(
        valid_raw_datapoint_count(event)
        for event in nda_result.selected
    )

    summary_path = (
        OUTPUT_DIR / "resumen_subconjunto.txt"
    )

    summary_lines = [
        "SUBCONJUNTO REDUCIDO OSDB V1.11",
        "=" * 72,
        f"Semilla aleatoria: {RANDOM_SEED}",
        "",
        "EVENTOS SELECCIONADOS",
        f"Tonic-Clonic: {len(tonic_clonic_result.selected)}",
        f"False Alarm: {len(false_alarm_result.selected)}",
        f"NDA: {len(nda_result.selected)}",
        "",
        "DATAPOINTS VÁLIDOS rawData[125]",
        f"Tonic-Clonic: {positive_windows}",
        f"False Alarm: {negative_windows_false}",
        f"NDA: {negative_windows_nda}",
        f"Negativos totales: "
        f"{negative_windows_false + negative_windows_nda}",
        "",
        "EVENTOS DISPONIBLES QUE CUMPLÍAN EL FILTRO",
        f"Tonic-Clonic: "
        f"{tonic_clonic_result.matching_events_seen}",
        f"False Alarm: "
        f"{false_alarm_result.matching_events_seen}",
        f"NDA: {nda_result.matching_events_seen}",
        "",
        "ARCHIVOS GENERADOS",
        str(output_seizures),
        str(output_false_alarms),
        str(output_nda),
        str(manifest_path),
    ]

    summary_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("")
    print("=" * 72)
    print("SUBCONJUNTO CREADO CORRECTAMENTE")
    print("=" * 72)
    print(f"Carpeta: {OUTPUT_DIR}")
    print(f"Resumen: {summary_path}")
    print(f"Manifiesto: {manifest_path}")
    print("")
    print("Datapoints válidos:")
    print(f"  Positivos Tonic-Clonic: {positive_windows}")
    print(
        "  Negativos False Alarm: "
        f"{negative_windows_false}"
    )
    print(f"  Negativos NDA: {negative_windows_nda}")


if __name__ == "__main__":
    main()

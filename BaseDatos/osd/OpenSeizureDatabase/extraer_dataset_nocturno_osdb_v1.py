"""
extraer_dataset_nocturno_osdb_v1.py
===================================

Extrae únicamente los eventos nocturnos seleccionados desde los JSON
completos de OSDB.

Entradas:
C:\\Users\\cesar\\osd\\dataset_nocturno_osdb_v1\\seleccion_final_v1\\
    eventIds_seleccion_nocturna_v1.json

C:\\Users\\cesar\\osd\\osdb_full_v1.11\\
    osdb_3min_allSeizures.json
    osdb_3min_falseAlarms.json

Salida:
C:\\Users\\cesar\\osd\\dataset_nocturno_osdb_v1\\dataset_extraido_v1\\

No divide TRAIN/VALIDATION/TEST.
No entrena.
No usa TEST.
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Iterator


FULL_DIR = Path(r"C:\Users\cesar\osd\osdb_full_v1.11")

SELECTION_FILE = Path(
    r"C:\Users\cesar\osd\dataset_nocturno_osdb_v1"
    r"\seleccion_final_v1\eventIds_seleccion_nocturna_v1.json"
)

OUTPUT_DIR = Path(
    r"C:\Users\cesar\osd\dataset_nocturno_osdb_v1"
    r"\dataset_extraido_v1"
)

SEIZURE_JSON = FULL_DIR / "osdb_3min_allSeizures.json"
FALSE_ALARM_JSON = FULL_DIR / "osdb_3min_falseAlarms.json"

CHUNK_SIZE = 4 * 1024 * 1024


def normalize_id(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        head = text[:-2]

        if head.replace("-", "").isdigit():
            text = head

    return text


def iter_json_array(
    file_path: Path,
    chunk_size: int = CHUNK_SIZE,
) -> Iterator[dict[str, Any]]:
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

            while position < len(buffer):
                char = buffer[position]

                if char.isspace() or char == ",":
                    position += 1
                    continue

                break

            if position >= len(buffer):
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
                remaining = buffer[position:]
                chunk = handle.read(chunk_size)

                if not chunk:
                    raise

                buffer = remaining + chunk
                position = 0
                continue

            if isinstance(obj, dict):
                yield obj

            position = end_position


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


def normalize_seizure_times(
    value: Any,
) -> list[float] | None:
    direct = parse_two_finite_numbers(value)

    if direct is not None:
        return direct

    if value is None:
        return None

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

        numeric = parse_two_finite_numbers(parsed)

        if numeric is not None:
            return numeric

        break

    numbers = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        cleaned,
    )

    if len(numbers) >= 2:
        return parse_two_finite_numbers(numbers[:2])

    return None


def valid_rawdata_count(
    event: dict[str, Any],
) -> int:
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


def extract_selected(
    source_path: Path,
    wanted_ids: set[str],
    label: str,
) -> tuple[list[dict[str, Any]], set[str]]:
    found: list[dict[str, Any]] = []
    found_ids: set[str] = set()

    total = 0

    print("")
    print(f"Leyendo {source_path.name}...")

    for event in iter_json_array(source_path):
        total += 1

        if total % 500 == 0:
            print(
                f"  Leídos: {total}; "
                f"seleccionados encontrados: {len(found_ids)}/"
                f"{len(wanted_ids)}",
                flush=True,
            )

        event_id = normalize_id(event.get("id"))

        if event_id not in wanted_ids:
            continue

        copied = dict(event)

        if label == "positive":
            copied["seizureTimes"] = normalize_seizure_times(
                copied.get("seizureTimes")
            )

        found.append(copied)
        found_ids.add(event_id)

        if found_ids == wanted_ids:
            # Ya encontramos todo lo requerido; no hace falta seguir
            # recorriendo un JSON grande.
            break

    missing = wanted_ids - found_ids

    return found, missing


def save_json(
    events: list[dict[str, Any]],
    path: Path,
) -> None:
    with path.open(
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


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        SELECTION_FILE,
        SEIZURE_JSON,
        FALSE_ALARM_JSON,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró:\n{path}"
            )

    selection = json.loads(
        SELECTION_FILE.read_text(
            encoding="utf-8"
        )
    )

    positive_ids = {
        normalize_id(value)
        for value in selection.get(
            "positive_event_ids",
            [],
        )
        if normalize_id(value)
    }

    negative_ids = {
        normalize_id(value)
        for value in selection.get(
            "negative_event_ids",
            [],
        )
        if normalize_id(value)
    }

    if not positive_ids:
        raise RuntimeError(
            "No hay positive_event_ids en la selección."
        )

    if not negative_ids:
        raise RuntimeError(
            "No hay negative_event_ids en la selección."
        )

    print("")
    print("=" * 78)
    print("EXTRACCIÓN DATASET NOCTURNO OSDB V1")
    print("=" * 78)
    print(f"Positivos solicitados: {len(positive_ids)}")
    print(f"Negativos solicitados: {len(negative_ids)}")

    positives, missing_positive = extract_selected(
        SEIZURE_JSON,
        positive_ids,
        "positive",
    )

    negatives, missing_negative = extract_selected(
        FALSE_ALARM_JSON,
        negative_ids,
        "negative",
    )

    positive_path = (
        OUTPUT_DIR / "osdb_nocturno_tonic_clonic.json"
    )

    negative_path = (
        OUTPUT_DIR / "osdb_nocturno_false_alarms.json"
    )

    save_json(positives, positive_path)
    save_json(negatives, negative_path)

    positive_windows = sum(
        valid_rawdata_count(event)
        for event in positives
    )

    negative_windows = sum(
        valid_rawdata_count(event)
        for event in negatives
    )

    positive_valid_seizure_times = sum(
        normalize_seizure_times(
            event.get("seizureTimes")
        )
        is not None
        for event in positives
    )

    summary_path = (
        OUTPUT_DIR
        / "resumen_dataset_nocturno_extraido_v1.txt"
    )

    lines = [
        "DATASET NOCTURNO OSDB V1 — EXTRACCIÓN",
        "=" * 78,
        "",
        "EVENTOS",
        f"Positivos solicitados: {len(positive_ids)}",
        f"Positivos encontrados: {len(positives)}",
        f"Negativos solicitados: {len(negative_ids)}",
        f"Negativos encontrados: {len(negatives)}",
        "",
        "DATAPOINTS rawData[125]",
        f"Positivos: {positive_windows}",
        f"Negativos: {negative_windows}",
        "",
        "SEIZURE TIMES",
        (
            "Positivos con seizureTimes válido: "
            f"{positive_valid_seizure_times}/{len(positives)}"
        ),
        "",
        "EVENTOS FALTANTES",
        (
            "Positivos: "
            + (
                ", ".join(sorted(missing_positive))
                if missing_positive
                else "ninguno"
            )
        ),
        (
            "Negativos: "
            + (
                ", ".join(sorted(missing_negative))
                if missing_negative
                else "ninguno"
            )
        ),
        "",
        "ARCHIVOS",
        str(positive_path),
        str(negative_path),
        "",
        "SIGUIENTE PASO",
        (
            "Si todos los eventos fueron encontrados y los positivos "
            "tienen seizureTimes válidos, preparar el pipeline nocturno "
            "TRAIN/VALIDATION/TEST."
        ),
        (
            "Todavía NO se ha dividido el dataset ni entrenado "
            "ningún modelo."
        ),
    ]

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("")
    print("=" * 78)
    print("EXTRACCIÓN COMPLETADA")
    print("=" * 78)
    print(
        f"Positivos encontrados: {len(positives)}/"
        f"{len(positive_ids)}"
    )
    print(
        f"Negativos encontrados: {len(negatives)}/"
        f"{len(negative_ids)}"
    )
    print(
        f"Ventanas rawData positivas: {positive_windows}"
    )
    print(
        f"Ventanas rawData negativas: {negative_windows}"
    )
    print(f"Resumen: {summary_path}")
    print("")
    print("Todavía NO se ha entrenado ningún modelo.")


if __name__ == "__main__":
    main()

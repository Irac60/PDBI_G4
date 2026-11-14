"""
crear_subconjunto_osdb_v2.py
============================

Crea un subconjunto OSDB ampliado y más útil para reducir falsas alarmas.

Selección:
- 120 eventos Seizure / Tonic-Clonic con rawData[125] y seizureTimes válidos.
- 40 falsas alarmas "difíciles" asociadas a movimiento.
- 20 falsas alarmas restantes/otros subtipos.
- 60 eventos NDA.

Balance por eventos:
- Positivos: 120
- Negativos: 40 + 20 + 60 = 120

Los JSON grandes se recorren progresivamente; no se cargan completos.

Salida:
C:\\Users\\cesar\\osd\\osdb_subset_v2

Archivos:
- osdb_3min_allSeizures.json
- osdb_3min_falseAlarms.json
- osdb_3min_ndaEvents.json
- manifiesto_subconjunto_v2.csv
- resumen_subconjunto_v2.txt

Ejecución:
    python .\\crear_subconjunto_osdb_v2.py
"""

from __future__ import annotations

import ast
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator


# ============================================================
# RUTAS Y PARÁMETROS
# ============================================================

DATA_DIR = Path(r"C:\Users\cesar\osd\osdb_full_v1.11")
OUTPUT_DIR = Path(r"C:\Users\cesar\osd\osdb_subset_v2")

RANDOM_SEED = 20261114

N_TONIC_CLONIC = 120
N_FALSE_ALARM_HARD = 40
N_FALSE_ALARM_OTHER = 20
N_NDA = 60

CHUNK_SIZE = 4 * 1024 * 1024

HARD_FALSE_ALARM_SUBTYPES = {
    "walking/running/cycling",
    "cooking/washing/cleaning",
    "typing/hand tools",
    "brushing hair/teeth",
    "motor vehicle",
    "typing",
    "sorting",
    "computer games",
    "sorting/knitting",
    "washing / cleaning",
    "talking",
    "pushing pram/wheelchair/lawn mower",
    "brushing teeth",
    "cycling",
    "walking",
    "pushing pram/wheelchair",
    "knitting",
    "cycle/run/walk",
    "washing up",
    "brushing hair",
}


# ============================================================
# LECTOR INCREMENTAL
# ============================================================

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

            if not isinstance(obj, dict):
                raise ValueError(
                    f"Se esperaba un objeto en {file_path.name}."
                )

            yield obj
            position = end_position


# ============================================================
# VALIDACIÓN
# ============================================================

def valid_raw_datapoint_count(
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

        if (
            isinstance(raw_data, list)
            and len(raw_data) >= 125
        ):
            count += 1

    return count


def parse_seizure_times(
    value: Any,
) -> list[float] | None:
    def validate(candidate: Any) -> list[float] | None:
        if (
            not isinstance(candidate, (list, tuple))
            or len(candidate) < 2
        ):
            return None

        try:
            start = float(candidate[0])
            end = float(candidate[1])
        except (TypeError, ValueError):
            return None

        if not (
            math.isfinite(start)
            and math.isfinite(end)
            and start < end
        ):
            return None

        return [start, end]

    direct = validate(value)

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

        validated = validate(parsed)

        if validated is not None:
            return validated

        break

    numbers = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        cleaned,
    )

    if len(numbers) >= 2:
        return validate(numbers[:2])

    return None


def is_tonic_clonic(
    event: dict[str, Any],
) -> bool:
    return (
        str(event.get("type", "")).strip().lower()
        == "seizure"
        and
        str(event.get("subType", "")).strip().lower()
        == "tonic-clonic"
        and
        valid_raw_datapoint_count(event) > 0
        and
        parse_seizure_times(event.get("seizureTimes"))
        is not None
    )


def false_alarm_group(
    event: dict[str, Any],
) -> str | None:
    if (
        str(event.get("type", "")).strip().lower()
        != "false alarm"
        or
        valid_raw_datapoint_count(event) == 0
    ):
        return None

    subtype = (
        str(event.get("subType", ""))
        .strip()
        .lower()
    )

    if subtype in HARD_FALSE_ALARM_SUBTYPES:
        return "hard"

    return "other"


def is_nda(
    event: dict[str, Any],
) -> bool:
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
class Reservoir:
    size: int
    seed: int

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.items: list[dict[str, Any]] = []
        self.matching_seen = 0

    def consider(
        self,
        event: dict[str, Any],
    ) -> None:
        self.matching_seen += 1

        if len(self.items) < self.size:
            self.items.append(event)
            return

        replacement = self.rng.randrange(
            self.matching_seen
        )

        if replacement < self.size:
            self.items[replacement] = event


def sample_single_category(
    file_path: Path,
    predicate: Callable[[dict[str, Any]], bool],
    size: int,
    seed: int,
    name: str,
) -> tuple[list[dict[str, Any]], int, int]:
    reservoir = Reservoir(size=size, seed=seed)
    total = 0

    print("")
    print(f"Procesando {name}: {file_path.name}")

    for event in iter_json_array(file_path):
        total += 1

        if total % 500 == 0:
            print(
                f"  Leídos: {total}; "
                f"válidos: {reservoir.matching_seen}",
                flush=True,
            )

        if predicate(event):
            reservoir.consider(event)

    if reservoir.matching_seen < size:
        raise RuntimeError(
            f"{name}: solo hay "
            f"{reservoir.matching_seen} eventos válidos; "
            f"se solicitaron {size}."
        )

    print(
        f"  Final: {total} leídos; "
        f"{reservoir.matching_seen} válidos; "
        f"{len(reservoir.items)} seleccionados."
    )

    return (
        reservoir.items,
        reservoir.matching_seen,
        total,
    )


def sample_false_alarm_groups(
    file_path: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
    int,
    int,
]:
    hard = Reservoir(
        size=N_FALSE_ALARM_HARD,
        seed=RANDOM_SEED + 2,
    )

    other = Reservoir(
        size=N_FALSE_ALARM_OTHER,
        seed=RANDOM_SEED + 3,
    )

    total = 0

    print("")
    print(
        "Procesando falsas alarmas difíciles y otras: "
        f"{file_path.name}"
    )

    for event in iter_json_array(file_path):
        total += 1

        if total % 500 == 0:
            print(
                f"  Leídos: {total}; "
                f"hard válidos: {hard.matching_seen}; "
                f"otros válidos: {other.matching_seen}",
                flush=True,
            )

        group = false_alarm_group(event)

        if group == "hard":
            hard.consider(event)
        elif group == "other":
            other.consider(event)

    if hard.matching_seen < N_FALSE_ALARM_HARD:
        raise RuntimeError(
            "No hay suficientes falsas alarmas difíciles."
        )

    if other.matching_seen < N_FALSE_ALARM_OTHER:
        raise RuntimeError(
            "No hay suficientes falsas alarmas restantes."
        )

    print(
        f"  Final: {total} leídos; "
        f"hard={hard.matching_seen}; "
        f"otros={other.matching_seen}."
    )

    return (
        hard.items,
        other.items,
        hard.matching_seen,
        other.matching_seen,
        total,
    )


# ============================================================
# GUARDADO
# ============================================================

def normalize_selected_event(
    event: dict[str, Any],
) -> dict[str, Any]:
    copied = dict(event)

    if (
        str(copied.get("type", "")).strip().lower()
        == "seizure"
    ):
        parsed = parse_seizure_times(
            copied.get("seizureTimes")
        )

        copied["seizureTimes"] = parsed

    return copied


def save_json(
    events: list[dict[str, Any]],
    path: Path,
) -> None:
    normalized = [
        normalize_selected_event(event)
        for event in events
    ]

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            normalized,
            handle,
            ensure_ascii=False,
            allow_nan=True,
            separators=(",", ":"),
        )


def manifest_rows(
    category: str,
    file_name: str,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for event in events:
        datapoints = event.get("datapoints")

        rows.append(
            {
                "selection_category": category,
                "source_file": file_name,
                "event_id": event.get("id"),
                "user_id": event.get("userId"),
                "type": event.get("type"),
                "subtype": event.get("subType"),
                "data_source": event.get(
                    "dataSourceName"
                ),
                "total_datapoints": (
                    len(datapoints)
                    if isinstance(datapoints, list)
                    else 0
                ),
                "valid_rawdata_125": (
                    valid_raw_datapoint_count(event)
                ),
                "seizure_times": repr(
                    parse_seizure_times(
                        event.get("seizureTimes")
                    )
                ),
            }
        )

    return rows


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    seizure_path = (
        DATA_DIR / "osdb_3min_allSeizures.json"
    )

    false_alarm_path = (
        DATA_DIR / "osdb_3min_falseAlarms.json"
    )

    nda_path = (
        DATA_DIR / "osdb_3min_ndaEvents.json"
    )

    for path in [
        seizure_path,
        false_alarm_path,
        nda_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"No existe:\n{path}"
            )

    tonic_clonic, tc_available, _ = (
        sample_single_category(
            file_path=seizure_path,
            predicate=is_tonic_clonic,
            size=N_TONIC_CLONIC,
            seed=RANDOM_SEED + 1,
            name="Tonic-Clonic con seizureTimes válido",
        )
    )

    (
        hard_false_alarms,
        other_false_alarms,
        hard_available,
        other_available,
        _,
    ) = sample_false_alarm_groups(
        false_alarm_path
    )

    nda_events, nda_available, _ = (
        sample_single_category(
            file_path=nda_path,
            predicate=is_nda,
            size=N_NDA,
            seed=RANDOM_SEED + 4,
            name="NDA",
        )
    )

    false_alarm_events = (
        hard_false_alarms
        + other_false_alarms
    )

    output_seizures = (
        OUTPUT_DIR
        / "osdb_3min_allSeizures.json"
    )

    output_false_alarms = (
        OUTPUT_DIR
        / "osdb_3min_falseAlarms.json"
    )

    output_nda = (
        OUTPUT_DIR
        / "osdb_3min_ndaEvents.json"
    )

    save_json(
        tonic_clonic,
        output_seizures,
    )

    save_json(
        false_alarm_events,
        output_false_alarms,
    )

    save_json(
        nda_events,
        output_nda,
    )

    rows: list[dict[str, Any]] = []

    rows.extend(
        manifest_rows(
            "tonic_clonic",
            output_seizures.name,
            tonic_clonic,
        )
    )

    rows.extend(
        manifest_rows(
            "false_alarm_hard",
            output_false_alarms.name,
            hard_false_alarms,
        )
    )

    rows.extend(
        manifest_rows(
            "false_alarm_other",
            output_false_alarms.name,
            other_false_alarms,
        )
    )

    rows.extend(
        manifest_rows(
            "nda",
            output_nda.name,
            nda_events,
        )
    )

    manifest_path = (
        OUTPUT_DIR
        / "manifiesto_subconjunto_v2.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    windows_tc = sum(
        valid_raw_datapoint_count(event)
        for event in tonic_clonic
    )

    windows_fa_hard = sum(
        valid_raw_datapoint_count(event)
        for event in hard_false_alarms
    )

    windows_fa_other = sum(
        valid_raw_datapoint_count(event)
        for event in other_false_alarms
    )

    windows_nda = sum(
        valid_raw_datapoint_count(event)
        for event in nda_events
    )

    summary_path = (
        OUTPUT_DIR
        / "resumen_subconjunto_v2.txt"
    )

    lines = [
        "SUBCONJUNTO AMPLIADO OSDB V2",
        "=" * 78,
        f"Semilla: {RANDOM_SEED}",
        "",
        "EVENTOS SELECCIONADOS",
        f"Tonic-Clonic: {len(tonic_clonic)}",
        f"False Alarm difíciles: {len(hard_false_alarms)}",
        f"False Alarm otras: {len(other_false_alarms)}",
        f"NDA: {len(nda_events)}",
        f"Positivos totales: {len(tonic_clonic)}",
        f"Negativos totales: "
        f"{len(false_alarm_events) + len(nda_events)}",
        "",
        "DATAPOINTS rawData[125]",
        f"Tonic-Clonic: {windows_tc}",
        f"False Alarm difíciles: {windows_fa_hard}",
        f"False Alarm otras: {windows_fa_other}",
        f"NDA: {windows_nda}",
        f"Negativos totales: "
        f"{windows_fa_hard + windows_fa_other + windows_nda}",
        "",
        "EVENTOS DISPONIBLES QUE CUMPLÍAN EL FILTRO",
        f"Tonic-Clonic con seizureTimes válido: {tc_available}",
        f"False Alarm difíciles: {hard_available}",
        f"False Alarm otras: {other_available}",
        f"NDA: {nda_available}",
        "",
        "ARCHIVOS",
        str(output_seizures),
        str(output_false_alarms),
        str(output_nda),
        str(manifest_path),
    ]

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("")
    print("=" * 78)
    print("SUBCONJUNTO V2 CREADO")
    print("=" * 78)
    print(f"Carpeta: {OUTPUT_DIR}")
    print(f"Resumen: {summary_path}")
    print(f"Manifiesto: {manifest_path}")


if __name__ == "__main__":
    main()

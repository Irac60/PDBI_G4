"""
buscar_crisis_nocturnas_osdb.py
===============================

Busca eventos POSITIVOS Tonic-Clonic con evidencia explícita de sueño/cama/reposo
en el CSV resumen de OSDB.

No usa la hora como evidencia de sueño.
No entrena modelos.
No toca TEST.

Entrada:
C:\\Users\\cesar\\osd\\osdb_full_v1.11\\osdb_3min_allSeizures.csv

Salida:
C:\\Users\\cesar\\osd\\dataset_nocturno_osdb_v1\\seleccion_candidatos
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DATA_DIR = Path(r"C:\Users\cesar\osd\osdb_full_v1.11")
CSV_PATH = DATA_DIR / "osdb_3min_allSeizures.csv"

OUTPUT_DIR = Path(
    r"C:\Users\cesar\osd\dataset_nocturno_osdb_v1"
    r"\seleccion_candidatos"
)

SLEEP_PATTERNS = {
    "sleep": [
        r"\bsleep\b",
        r"\bsleeping\b",
        r"\basleep\b",
        r"\bbedtime\b",
        r"\bovernight\b",
        r"\bnap\b",
        r"\bnapping\b",
        r"\bdozing\b",
        r"\bdormir\b",
        r"\bdurmiendo\b",
        r"\bdormid[oa]\b",
        r"\bsiesta\b",
    ],
    "bed": [
        r"\bin bed\b",
        r"\bbed\b",
        r"\blying in bed\b",
        r"\blaying in bed\b",
        r"\bacostad[oa]\b",
        r"\ben la cama\b",
        r"\bcama\b",
    ],
    "rest": [
        r"\bresting\b",
        r"\bat rest\b",
        r"\brest\b",
        r"\blying down\b",
        r"\blaying down\b",
        r"\breposo\b",
        r"\bdescansando\b",
        r"\bdescanso\b",
    ],
}


def first_existing_column(
    df: pd.DataFrame,
    names: list[str],
) -> str | None:
    normalized = {
        str(c).strip().lower(): str(c)
        for c in df.columns
    }

    for name in names:
        key = name.strip().lower()

        if key in normalized:
            return normalized[key]

    return None


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def match_groups(
    text: str,
    groups: dict[str, list[str]],
) -> list[str]:
    output: list[str] = []
    lowered = text.lower()

    for group_name, patterns in groups.items():
        if any(
            re.search(pattern, lowered, flags=re.IGNORECASE)
            for pattern in patterns
        ):
            output.append(group_name)

    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró:\n{CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH, low_memory=False)

    event_col = first_existing_column(
        df, ["eventId", "id", "event_id"]
    )
    user_col = first_existing_column(
        df, ["userId", "user_id"]
    )
    time_col = first_existing_column(
        df, ["dataTime", "datetime", "dateTime"]
    )
    type_col = first_existing_column(
        df, ["type"]
    )
    subtype_col = first_existing_column(
        df, ["subType", "subtype"]
    )
    desc_col = first_existing_column(
        df, ["desc", "description", "notes", "note"]
    )

    if event_col is None or subtype_col is None or desc_col is None:
        raise RuntimeError(
            "Faltan columnas esenciales eventId/subType/desc."
        )

    type_series = (
        df[type_col].fillna("").astype(str).str.strip().str.lower()
        if type_col
        else pd.Series([""] * len(df))
    )

    subtype_series = (
        df[subtype_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Conservador: solo Tonic-Clonic.
    mask_tc = subtype_series.eq("tonic-clonic")

    # Si existe type, exigir Seizure.
    if type_col:
        mask_tc &= type_series.eq("seizure")

    tc = df.loc[mask_tc].copy()

    rows: list[dict[str, Any]] = []

    for _, row in tc.iterrows():
        desc = normalize_text(row[desc_col])
        subtype = normalize_text(row[subtype_col])

        combined = " | ".join(
            item for item in [desc, subtype] if item
        )

        sleep_matches = match_groups(
            combined,
            SLEEP_PATTERNS,
        )

        if not sleep_matches:
            continue

        rows.append(
            {
                "eventId": row[event_col],
                "userId": (
                    row[user_col]
                    if user_col
                    else ""
                ),
                "dataTime": (
                    row[time_col]
                    if time_col
                    else ""
                ),
                "type": (
                    row[type_col]
                    if type_col
                    else "Seizure"
                ),
                "subType": row[subtype_col],
                "desc": desc,
                "sleep_evidence": ";".join(
                    sleep_matches
                ),
                "source_csv": CSV_PATH.name,
            }
        )

    result = pd.DataFrame(rows)

    candidates_path = (
        OUTPUT_DIR
        / "crisis_tonic_clonic_nocturnas_explicitas.csv"
    )

    ids_path = (
        OUTPUT_DIR
        / "eventIds_crisis_nocturnas.json"
    )

    summary_path = (
        OUTPUT_DIR
        / "resumen_busqueda_crisis_nocturnas.txt"
    )

    result.to_csv(
        candidates_path,
        index=False,
        encoding="utf-8-sig",
    )

    ids = (
        [str(v) for v in result["eventId"].dropna().tolist()]
        if not result.empty
        else []
    )

    ids_path.write_text(
        json.dumps(
            ids,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "BÚSQUEDA DE CRISIS TÓNICO-CLÓNICAS NOCTURNAS EN OSDB",
        "=" * 78,
        "",
        "CRITERIO",
        "SubType = Tonic-Clonic.",
        "Se exige evidencia explícita de sueño/cama/reposo en desc/subType.",
        "La hora del evento NO se usa como evidencia de sueño.",
        "",
        f"Eventos totales en allSeizures CSV: {len(df)}",
        f"Eventos Tonic-Clonic totales: {len(tc)}",
        f"Tonic-Clonic con evidencia explícita nocturna: {len(result)}",
    ]

    if not result.empty:
        lines.extend(
            [
                f"Usuarios distintos: "
                f"{result['userId'].astype(str).nunique()}",
                "",
                "EVIDENCIA DE SUEÑO/REPOSO",
                result["sleep_evidence"]
                .value_counts(dropna=False)
                .to_string(),
                "",
                "PRIMEROS 30 CANDIDATOS",
                result[
                    [
                        "eventId",
                        "userId",
                        "dataTime",
                        "subType",
                        "sleep_evidence",
                        "desc",
                    ]
                ]
                .head(30)
                .to_string(index=False),
            ]
        )
    else:
        lines.extend(
            [
                "",
                "No se encontraron suficientes positivos nocturnos explícitos.",
                (
                    "En ese caso el siguiente paso será conservar los "
                    "Tonic-Clonic generales como positivos y documentar "
                    "esa limitación de dominio."
                ),
            ]
        )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("")
    print("=" * 78)
    print("BÚSQUEDA DE POSITIVOS NOCTURNOS COMPLETADA")
    print("=" * 78)
    print(f"Tonic-Clonic totales: {len(tc)}")
    print(
        "Tonic-Clonic nocturnas explícitas: "
        f"{len(result)}"
    )
    print(f"Resumen: {summary_path}")
    print("")
    print("Todavía NO se ha entrenado ningún modelo.")


if __name__ == "__main__":
    main()

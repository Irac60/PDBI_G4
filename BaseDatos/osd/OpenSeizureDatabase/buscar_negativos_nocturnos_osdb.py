"""
buscar_negativos_nocturnos_osdb.py
==================================

Busca candidatos NEGATIVOS NOCTURNOS en los archivos resumen CSV de OSDB.

Criterio principal:
- El texto del evento debe mencionar explícitamente sueño, cama o reposo.
- Se excluyen descripciones que indiquen actividad fuera de cama o despertar,
  porque el proyecto actual se limita a dormido/descansando.

IMPORTANTE:
- La hora del evento NO se usa como prueba de sueño.
- Este script SOLO IDENTIFICA candidatos; todavía NO entrena ni abre TEST.
- Después revisaremos cuántos candidatos existen antes de extraer los JSON.

Entrada esperada:
C:\\Users\\cesar\\osd\\osdb_full_v1.11

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

OUTPUT_DIR = Path(
    r"C:\Users\cesar\osd\dataset_nocturno_osdb_v1"
    r"\seleccion_candidatos"
)

CSV_CANDIDATES = [
    DATA_DIR / "osdb_3min_falseAlarms.csv",
    DATA_DIR / "osdb_3min_ndaEvents.csv",
]

# Expresiones que sí justifican considerar el evento como sueño/reposo.
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
        r"\bdozed\b",
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

# Por ahora el usuario pidió ignorar levantarse/despertarse y actividad diurna.
EXCLUDE_PATTERNS = {
    "walking_running": [
        r"\bwalk(?:ing|ed)?\b",
        r"\brun(?:ning)?\b",
        r"\bjog(?:ging)?\b",
        r"\bcamin(?:ar|ando|é|aba)?\b",
        r"\bcorrer\b",
        r"\bcorriendo\b",
    ],
    "cycling_vehicle": [
        r"\bcycl(?:e|ing)\b",
        r"\bbike\b",
        r"\bbicycle\b",
        r"\bdriv(?:e|ing)\b",
        r"\bcar\b",
        r"\bvehicle\b",
        r"\bbicicleta\b",
        r"\bmanejando\b",
    ],
    "daily_activity": [
        r"\bcook(?:ing|ed)?\b",
        r"\bclean(?:ing|ed)?\b",
        r"\btyping\b",
        r"\btool(?:s)?\b",
        r"\bbrushing\b",
        r"\bteeth\b",
        r"\bhair\b",
        r"\bexercise\b",
        r"\bgym\b",
        r"\bshopping\b",
        r"\bcocinando\b",
        r"\blimpiando\b",
        r"\bescribiendo\b",
        r"\bejercicio\b",
    ],
    "awakening_getting_up": [
        r"\bwoke\b",
        r"\bwoken\b",
        r"\bwaking\b",
        r"\bawake\b",
        r"\bgetting up\b",
        r"\bgot up\b",
        r"\bget out of bed\b",
        r"\bout of bed\b",
        r"\bbathroom\b",
        r"\btoilet\b",
        r"\bshower\b",
        r"\bstairs\b",
        r"\bdespert(?:é|o|ando|ado|ada)\b",
        r"\blevant(?:é|arse|ando|ado|ada)\b",
        r"\bbaño\b",
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
            re.search(
                pattern,
                lowered,
                flags=re.IGNORECASE,
            )
            for pattern in patterns
        ):
            output.append(group_name)

    return output


def load_summary_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró:\n{path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    event_col = first_existing_column(
        df,
        ["eventId", "id", "event_id"],
    )

    user_col = first_existing_column(
        df,
        ["userId", "user_id"],
    )

    time_col = first_existing_column(
        df,
        ["dataTime", "datetime", "dateTime"],
    )

    type_col = first_existing_column(
        df,
        ["type"],
    )

    subtype_col = first_existing_column(
        df,
        ["subType", "subtype"],
    )

    desc_col = first_existing_column(
        df,
        ["desc", "description", "notes", "note"],
    )

    if event_col is None:
        raise RuntimeError(
            f"{path.name}: no encontré columna eventId/id."
        )

    if desc_col is None:
        raise RuntimeError(
            f"{path.name}: no encontré columna desc/description."
        )

    out = pd.DataFrame(
        {
            "eventId": df[event_col],
            "userId": (
                df[user_col]
                if user_col
                else ""
            ),
            "dataTime": (
                df[time_col]
                if time_col
                else ""
            ),
            "type": (
                df[type_col]
                if type_col
                else ""
            ),
            "subType": (
                df[subtype_col]
                if subtype_col
                else ""
            ),
            "desc": df[desc_col],
            "source_csv": path.name,
        }
    )

    return out


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = [
        load_summary_csv(path)
        for path in CSV_CANDIDATES
    ]

    events = pd.concat(
        frames,
        ignore_index=True,
    )

    rows: list[dict[str, Any]] = []

    for _, row in events.iterrows():
        desc = normalize_text(
            row["desc"]
        )

        subtype = normalize_text(
            row["subType"]
        )

        combined = " | ".join(
            item
            for item in [desc, subtype]
            if item
        )

        sleep_matches = match_groups(
            combined,
            SLEEP_PATTERNS,
        )

        exclude_matches = match_groups(
            combined,
            EXCLUDE_PATTERNS,
        )

        if sleep_matches:
            status = (
                "candidate"
                if not exclude_matches
                else "excluded"
            )

            rows.append(
                {
                    "status": status,
                    "eventId": row["eventId"],
                    "userId": row["userId"],
                    "dataTime": row["dataTime"],
                    "type": row["type"],
                    "subType": row["subType"],
                    "desc": desc,
                    "sleep_evidence": ";".join(
                        sleep_matches
                    ),
                    "exclude_reason": ";".join(
                        exclude_matches
                    ),
                    "source_csv": row["source_csv"],
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        candidates = result.copy()
        excluded = result.copy()
    else:
        candidates = result.loc[
            result["status"] == "candidate"
        ].copy()

        excluded = result.loc[
            result["status"] == "excluded"
        ].copy()

    candidates_path = (
        OUTPUT_DIR
        / "candidatos_nocturnos_explicitos.csv"
    )

    excluded_path = (
        OUTPUT_DIR
        / "candidatos_excluidos_por_actividad.csv"
    )

    result_path = (
        OUTPUT_DIR
        / "todos_eventos_con_palabras_nocturnas.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "resumen_busqueda_negativos_nocturnos.txt"
    )

    ids_path = (
        OUTPUT_DIR
        / "eventIds_candidatos_nocturnos.json"
    )

    result.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig",
    )

    candidates.to_csv(
        candidates_path,
        index=False,
        encoding="utf-8-sig",
    )

    excluded.to_csv(
        excluded_path,
        index=False,
        encoding="utf-8-sig",
    )

    candidate_ids = [
        str(value)
        for value in candidates.get(
            "eventId",
            pd.Series(dtype=object),
        ).dropna().tolist()
    ]

    ids_path.write_text(
        json.dumps(
            candidate_ids,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Resúmenes por fuente/tipo y evidencia.
    lines = [
        "BÚSQUEDA DE NEGATIVOS NOCTURNOS EN OSDB",
        "=" * 78,
        "",
        "CRITERIO",
        (
            "Solo se aceptan eventos cuyo desc/subType contenga "
            "evidencia explícita de sueño, cama o reposo."
        ),
        (
            "La hora del evento NO se utiliza como evidencia "
            "de que la persona estaba dormida."
        ),
        (
            "Se excluyen por ahora caminar/correr, actividad diurna "
            "y eventos que describen despertar/levantarse."
        ),
        "",
        f"Eventos revisados en CSV: {len(events)}",
        f"Eventos con palabras de sueño/reposo: {len(result)}",
        f"Candidatos nocturnos aceptados: {len(candidates)}",
        f"Excluidos por actividad incompatible: {len(excluded)}",
        "",
    ]

    if not candidates.empty:
        lines.extend(
            [
                "CANDIDATOS POR ARCHIVO",
                candidates[
                    "source_csv"
                ].value_counts(
                    dropna=False
                ).to_string(),
                "",
                "CANDIDATOS POR TIPO",
                candidates[
                    "type"
                ].fillna("<VACÍO>").value_counts(
                    dropna=False
                ).to_string(),
                "",
                "EVIDENCIA DE SUEÑO/REPOSO",
                candidates[
                    "sleep_evidence"
                ].value_counts(
                    dropna=False
                ).to_string(),
                "",
                "USUARIOS DISTINTOS",
                str(
                    candidates[
                        "userId"
                    ].astype(str).nunique()
                ),
                "",
                "PRIMEROS 25 CANDIDATOS",
                candidates[
                    [
                        "eventId",
                        "userId",
                        "dataTime",
                        "type",
                        "subType",
                        "sleep_evidence",
                        "desc",
                    ]
                ].head(25).to_string(
                    index=False
                ),
            ]
        )
    else:
        lines.extend(
            [
                "No se encontraron candidatos explícitos.",
                (
                    "Si esto ocurre, NO ampliaremos el filtro "
                    "por hora automáticamente; primero revisaremos "
                    "qué texto existe realmente en OSDB."
                ),
            ]
        )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("")
    print("=" * 78)
    print("BÚSQUEDA NOCTURNA COMPLETADA")
    print("=" * 78)
    print(f"Eventos revisados: {len(events)}")
    print(
        "Con evidencia sueño/reposo: "
        f"{len(result)}"
    )
    print(
        "Candidatos aceptados: "
        f"{len(candidates)}"
    )
    print(
        "Excluidos: "
        f"{len(excluded)}"
    )
    print(f"Resumen: {summary_path}")
    print("")
    print(
        "Todavía NO se ha entrenado ningún modelo "
        "ni se ha utilizado TEST."
    )


if __name__ == "__main__":
    main()

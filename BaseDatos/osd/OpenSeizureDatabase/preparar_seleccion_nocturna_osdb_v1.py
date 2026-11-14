"""
preparar_seleccion_nocturna_osdb_v1.py
======================================

Construye una selección preliminar NOCTURNA usando únicamente los CSV de
candidatos ya generados.

Objetivo:
- conservar Tonic-Clonic con contexto explícito de sueño/cama;
- excluir positivos claramente fuera del alcance (levantarse/caminar/estar fuera de cama);
- usar negativos nocturnos de los mismos usuarios que los positivos;
- evitar duplicados obvios asociados a eventos fusionados;
- seleccionar hasta 2 negativos por cada positivo de cada usuario;
- NO leer todavía los JSON gigantes;
- NO entrenar;
- NO usar TEST.

Entradas:
C:\\Users\\cesar\\osd\\dataset_nocturno_osdb_v1\\seleccion_candidatos\\
    crisis_tonic_clonic_nocturnas_explicitas.csv
    candidatos_nocturnos_explicitos.csv

Salidas:
C:\\Users\\cesar\\osd\\dataset_nocturno_osdb_v1\\seleccion_final_v1\\
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(r"C:\Users\cesar\osd\dataset_nocturno_osdb_v1")
CANDIDATE_DIR = ROOT / "seleccion_candidatos"
OUTPUT_DIR = ROOT / "seleccion_final_v1"

POSITIVE_PATH = (
    CANDIDATE_DIR
    / "crisis_tonic_clonic_nocturnas_explicitas.csv"
)

NEGATIVE_PATH = (
    CANDIDATE_DIR
    / "candidatos_nocturnos_explicitos.csv"
)

RANDOM_SEED = 20261114
NEGATIVE_RATIO = 2  # hasta 2 negativos por positivo del mismo usuario


# Expresiones que implican que el evento positivo sale del alcance actual:
# dormido/descansando, sin analizar levantarse/caminar.
OUT_OF_SCOPE_POSITIVE_PATTERNS = [
    r"\bgot out of bed\b",
    r"\bget out of bed\b",
    r"\bout of bed\b",
    r"\bstood up\b",
    r"\bwalking\b",
    r"\bwalked\b",
    r"\bcircled\b",
    r"\blevant(?:arse|ó|ado|ada)\b",
    r"\bcamin(?:ó|ando|ar)\b",
]

# Solo se marcan, NO se eliminan automáticamente.
UNCERTAIN_POSITIVE_PATTERNS = [
    r"\bassum(?:e|ing|ed)\b",
    r"\?\s*seizure\b",
    r"\?\s*trigger\b",
    r"\bpossible\b",
    r"\bsuspect(?:ed)?\b",
]


def normalize_id(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        head = text[:-2]

        if head.replace("-", "").isdigit():
            text = head

    return text


def normalize_user(value: Any) -> str:
    return normalize_id(value)


def has_any_pattern(
    text: str,
    patterns: list[str],
) -> bool:
    lowered = str(text).lower()

    return any(
        re.search(pattern, lowered, flags=re.IGNORECASE)
        for pattern in patterns
    )


def merged_event_ids(text: str) -> set[str]:
    """
    Extrae IDs que aparecen después de frases como:
    'Includes data from merged event(s): 6563'
    """
    value = str(text)

    ids: set[str] = set()

    patterns = [
        r"merged event\(s\)\s*:\s*([0-9,\s]+)",
        r"merged events?\s*:\s*([0-9,\s]+)",
    ]

    for pattern in patterns:
        for match in re.finditer(
            pattern,
            value,
            flags=re.IGNORECASE,
        ):
            for item in re.findall(
                r"\d+",
                match.group(1),
            ):
                ids.add(item)

    return ids


def clean_merged_duplicates(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Si un candidato A declara que incluye datos del evento B y B también
    aparece como candidato independiente, se conserva A y se marca B
    como posible duplicado fusionado.
    """
    all_ids = set(df["eventId_norm"])

    child_ids: set[str] = set()

    for _, row in df.iterrows():
        parent_id = row["eventId_norm"]

        for child_id in merged_event_ids(
            row.get("desc", "")
        ):
            if (
                child_id in all_ids
                and child_id != parent_id
            ):
                child_ids.add(child_id)

    duplicates = df.loc[
        df["eventId_norm"].isin(child_ids)
    ].copy()

    cleaned = df.loc[
        ~df["eventId_norm"].isin(child_ids)
    ].copy()

    return cleaned, duplicates


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        POSITIVE_PATH,
        NEGATIVE_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró:\n{path}"
            )

    positives = pd.read_csv(
        POSITIVE_PATH,
        low_memory=False,
    )

    negatives = pd.read_csv(
        NEGATIVE_PATH,
        low_memory=False,
    )

    for name, df in [
        ("positivos", positives),
        ("negativos", negatives),
    ]:
        if "eventId" not in df.columns:
            raise RuntimeError(
                f"{name}: falta eventId."
            )

        if "userId" not in df.columns:
            raise RuntimeError(
                f"{name}: falta userId."
            )

        if "desc" not in df.columns:
            df["desc"] = ""

    positives = positives.copy()
    negatives = negatives.copy()

    positives["eventId_norm"] = (
        positives["eventId"].map(normalize_id)
    )

    positives["userId_norm"] = (
        positives["userId"].map(normalize_user)
    )

    negatives["eventId_norm"] = (
        negatives["eventId"].map(normalize_id)
    )

    negatives["userId_norm"] = (
        negatives["userId"].map(normalize_user)
    )

    positives["fuera_alcance_actual"] = (
        positives["desc"]
        .fillna("")
        .astype(str)
        .map(
            lambda text: has_any_pattern(
                text,
                OUT_OF_SCOPE_POSITIVE_PATTERNS,
            )
        )
    )

    positives["nota_incertidumbre"] = (
        positives["desc"]
        .fillna("")
        .astype(str)
        .map(
            lambda text: has_any_pattern(
                text,
                UNCERTAIN_POSITIVE_PATTERNS,
            )
        )
    )

    excluded_positives = positives.loc[
        positives["fuera_alcance_actual"]
    ].copy()

    retained_positives = positives.loc[
        ~positives["fuera_alcance_actual"]
    ].copy()

    retained_positives, duplicate_positives = (
        clean_merged_duplicates(
            retained_positives
        )
    )

    positive_users = sorted(
        set(
            retained_positives[
                "userId_norm"
            ].astype(str)
        )
    )

    same_user_negatives = negatives.loc[
        negatives["userId_norm"].isin(
            positive_users
        )
    ].copy()

    same_user_negatives, duplicate_negatives = (
        clean_merged_duplicates(
            same_user_negatives
        )
    )

    # --------------------------------------------------------
    # Selección determinística de negativos: hasta 2:1 por usuario.
    # --------------------------------------------------------
    rng = random.Random(RANDOM_SEED)
    selected_negative_parts = []

    user_summary_rows = []

    for user_id in positive_users:
        pos_user = retained_positives.loc[
            retained_positives[
                "userId_norm"
            ] == user_id
        ].copy()

        neg_user = same_user_negatives.loc[
            same_user_negatives[
                "userId_norm"
            ] == user_id
        ].copy()

        requested = (
            len(pos_user) * NEGATIVE_RATIO
        )

        available = len(neg_user)

        selected_n = min(
            requested,
            available,
        )

        if selected_n > 0:
            indices = list(neg_user.index)
            rng.shuffle(indices)
            selected_indices = indices[
                :selected_n
            ]

            selected_negative_parts.append(
                neg_user.loc[
                    selected_indices
                ].copy()
            )

        user_summary_rows.append(
            {
                "userId": user_id,
                "positivos_nocturnos": len(
                    pos_user
                ),
                "negativos_nocturnos_disponibles": available,
                "negativos_solicitados_2a1": requested,
                "negativos_seleccionados": selected_n,
            }
        )

    if selected_negative_parts:
        selected_negatives = pd.concat(
            selected_negative_parts,
            ignore_index=True,
        )
    else:
        selected_negatives = (
            same_user_negatives.iloc[0:0].copy()
        )

    user_summary = pd.DataFrame(
        user_summary_rows
    )

    # --------------------------------------------------------
    # Manifiesto final.
    # --------------------------------------------------------
    pos_manifest = retained_positives.copy()
    pos_manifest["label"] = 1
    pos_manifest["class_name"] = (
        "Tonic-Clonic nocturna"
    )

    neg_manifest = selected_negatives.copy()
    neg_manifest["label"] = 0
    neg_manifest["class_name"] = (
        "False Alarm sueño/reposo"
    )

    shared_columns = sorted(
        set(pos_manifest.columns).union(
            set(neg_manifest.columns)
        )
    )

    pos_manifest = pos_manifest.reindex(
        columns=shared_columns
    )

    neg_manifest = neg_manifest.reindex(
        columns=shared_columns
    )

    final_manifest = pd.concat(
        [
            pos_manifest,
            neg_manifest,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Guardado.
    # --------------------------------------------------------
    retained_path = (
        OUTPUT_DIR
        / "positivos_nocturnos_retenidos.csv"
    )

    excluded_path = (
        OUTPUT_DIR
        / "positivos_excluidos_fuera_alcance.csv"
    )

    same_user_neg_path = (
        OUTPUT_DIR
        / "negativos_nocturnos_mismos_usuarios.csv"
    )

    selected_neg_path = (
        OUTPUT_DIR
        / "negativos_nocturnos_seleccionados.csv"
    )

    duplicates_pos_path = (
        OUTPUT_DIR
        / "duplicados_fusionados_positivos.csv"
    )

    duplicates_neg_path = (
        OUTPUT_DIR
        / "duplicados_fusionados_negativos.csv"
    )

    user_summary_path = (
        OUTPUT_DIR
        / "resumen_por_usuario.csv"
    )

    manifest_path = (
        OUTPUT_DIR
        / "manifiesto_seleccion_nocturna_v1.csv"
    )

    ids_path = (
        OUTPUT_DIR
        / "eventIds_seleccion_nocturna_v1.json"
    )

    summary_path = (
        OUTPUT_DIR
        / "resumen_seleccion_nocturna_v1.txt"
    )

    retained_positives.to_csv(
        retained_path,
        index=False,
        encoding="utf-8-sig",
    )

    excluded_positives.to_csv(
        excluded_path,
        index=False,
        encoding="utf-8-sig",
    )

    same_user_negatives.to_csv(
        same_user_neg_path,
        index=False,
        encoding="utf-8-sig",
    )

    selected_negatives.to_csv(
        selected_neg_path,
        index=False,
        encoding="utf-8-sig",
    )

    duplicate_positives.to_csv(
        duplicates_pos_path,
        index=False,
        encoding="utf-8-sig",
    )

    duplicate_negatives.to_csv(
        duplicates_neg_path,
        index=False,
        encoding="utf-8-sig",
    )

    user_summary.to_csv(
        user_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    final_manifest.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    ids_payload = {
        "positive_event_ids": (
            retained_positives[
                "eventId_norm"
            ].astype(str).tolist()
        ),
        "negative_event_ids": (
            selected_negatives[
                "eventId_norm"
            ].astype(str).tolist()
        ),
    }

    ids_path.write_text(
        json.dumps(
            ids_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    uncertain_count = int(
        retained_positives[
            "nota_incertidumbre"
        ].sum()
    )

    lines = [
        "SELECCIÓN NOCTURNA OSDB V1",
        "=" * 78,
        "",
        "ALCANCE",
        (
            "Solo eventos compatibles con dormido/descansando. "
            "Se excluyen positivos con descripción explícita de "
            "levantarse/caminar/estar fuera de cama."
        ),
        (
            "Los negativos se limitan a los mismos usuarios "
            "que aportan positivos nocturnos."
        ),
        (
            f"Se seleccionan hasta {NEGATIVE_RATIO} negativos "
            "por positivo de cada usuario."
        ),
        "",
        "POSITIVOS",
        f"Candidatos iniciales: {len(positives)}",
        f"Excluidos fuera del alcance: {len(excluded_positives)}",
        f"Duplicados fusionados excluidos: {len(duplicate_positives)}",
        f"Positivos retenidos: {len(retained_positives)}",
        f"Positivos retenidos con nota de incertidumbre: {uncertain_count}",
        "",
        "NEGATIVOS",
        f"Candidatos iniciales: {len(negatives)}",
        f"Negativos de los mismos usuarios positivos: "
        f"{len(same_user_negatives)}",
        f"Duplicados fusionados excluidos: {len(duplicate_negatives)}",
        f"Negativos seleccionados: {len(selected_negatives)}",
        "",
        "USUARIOS POSITIVOS",
        ", ".join(positive_users),
        "",
        "RESUMEN POR USUARIO",
        user_summary.to_string(index=False),
        "",
        "TOTAL SELECCIÓN",
        f"Positivos: {len(retained_positives)}",
        f"Negativos: {len(selected_negatives)}",
        f"Eventos totales: {len(final_manifest)}",
        "",
        "POSITIVOS EXCLUIDOS FUERA DEL ALCANCE",
        (
            excluded_positives[
                ["eventId", "userId", "desc"]
            ].to_string(index=False)
            if not excluded_positives.empty
            else "Ninguno."
        ),
        "",
        "POSITIVOS RETENIDOS CON NOTA DE INCERTIDUMBRE",
        (
            retained_positives.loc[
                retained_positives[
                    "nota_incertidumbre"
                ],
                ["eventId", "userId", "desc"],
            ].to_string(index=False)
            if uncertain_count > 0
            else "Ninguno."
        ),
        "",
        "SIGUIENTE PASO",
        (
            "Si esta selección es suficiente, extraer únicamente "
            "estos eventId desde los JSON completos de OSDB."
        ),
        (
            "Todavía NO se ha entrenado, dividido TRAIN/VALIDATION/TEST "
            "ni evaluado ningún modelo."
        ),
    ]

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("")
    print("=" * 78)
    print("SELECCIÓN NOCTURNA V1 PREPARADA")
    print("=" * 78)
    print(
        f"Positivos retenidos: "
        f"{len(retained_positives)}"
    )
    print(
        f"Negativos mismos usuarios disponibles: "
        f"{len(same_user_negatives)}"
    )
    print(
        f"Negativos seleccionados: "
        f"{len(selected_negatives)}"
    )
    print(
        f"Usuarios positivos: "
        f"{len(positive_users)}"
    )
    print(f"Resumen: {summary_path}")
    print("")
    print(
        "Todavía NO se han leído los JSON grandes ni entrenado un modelo."
    )


if __name__ == "__main__":
    main()

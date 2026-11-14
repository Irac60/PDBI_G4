"""
dividir_osdb_prototipo.py
=========================

Divide allData.csv por EVENTO usando la función oficial splitData de
nnTraining2 y verifica que no exista fuga del mismo eventId entre:

- trainData.csv
- valData.csv
- testData.csv

También genera:
- manifiesto_division_eventos.csv
- resumen_division_eventos.txt

IMPORTANTE
----------
Esta primera división es independiente por EVENTO, no necesariamente por
USUARIO. El resumen informa si un mismo userId aparece en varios conjuntos.

Ejecución desde la raíz del repositorio:
    python .\dividir_osdb_prototipo.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from user_tools.nnTraining2 import splitData


# ============================================================
# RUTAS
# ============================================================

REPO_DIR = Path(r"C:\Users\cesar\osd\OpenSeizureDatabase")
CONFIG_PATH = REPO_DIR / "nnConfig_prototipo_pytorch.json"
PIPELINE_DIR = Path(r"C:\Users\cesar\osd\pipeline_prototipo")

REGENERAR_DIVISION = True


# ============================================================
# FUNCIONES
# ============================================================

def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró la configuración:\n{path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_binary_label(value: Any) -> int:
    """
    Convierte la columna type a 0/1.

    flattenData normalmente genera:
      1 -> Seizure
      0 -> False Alarm / nda
    """
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

    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"No se pudo interpretar la etiqueta: {value!r}"
        )

    if number == 1:
        return 1

    if number == 0:
        return 0

    raise ValueError(
        f"Etiqueta fuera del formato binario 0/1: {value!r}"
    )


def load_split_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No se generó el archivo:\n{path}"
        )

    df = pd.read_csv(path, low_memory=False)

    required = {
        "eventId",
        "userId",
        "type",
        "typeStr",
    }

    missing = required.difference(df.columns)

    if missing:
        raise RuntimeError(
            f"{path.name} no contiene las columnas: "
            + ", ".join(sorted(missing))
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
    """
    Una fila por evento con su etiqueta, usuario y número de ventanas.
    """
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


def set_as_strings(series: pd.Series) -> set[str]:
    return set(
        series.astype(str).tolist()
    )


def overlap(
    first: pd.Series,
    second: pd.Series,
) -> set[str]:
    return (
        set_as_strings(first)
        .intersection(set_as_strings(second))
    )


def split_metrics(
    df: pd.DataFrame,
    event_df: pd.DataFrame,
) -> dict[str, Any]:
    magnitude_columns = [
        f"M{i:03d}"
        for i in range(125)
    ]

    available_magnitude = [
        column
        for column in magnitude_columns
        if column in df.columns
    ]

    complete_magnitude_rows = (
        int(
            df[available_magnitude]
            .notna()
            .all(axis=1)
            .sum()
        )
        if len(available_magnitude) == 125
        else 0
    )

    return {
        "rows": len(df),
        "events": len(event_df),
        "positive_rows": int(
            (df["_label"] == 1).sum()
        ),
        "negative_rows": int(
            (df["_label"] == 0).sum()
        ),
        "positive_events": int(
            (event_df["label"] == 1).sum()
        ),
        "negative_events": int(
            (event_df["label"] == 0).sum()
        ),
        "users": int(
            df["userId"]
            .astype(str)
            .nunique()
        ),
        "complete_magnitude_rows": (
            complete_magnitude_rows
        ),
    }


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:
    config = load_config(CONFIG_PATH)

    all_csv_name = config[
        "dataFileNames"
    ].get(
        "allDataFileCsv",
        "allData.csv",
    )

    train_csv_name = config[
        "dataFileNames"
    ].get(
        "trainDataFileCsv",
        "trainData.csv",
    )

    val_csv_name = config[
        "dataFileNames"
    ].get(
        "valDataFileCsv",
        "valData.csv",
    )

    test_csv_name = config[
        "dataFileNames"
    ].get(
        "testDataFileCsv",
        "testData.csv",
    )

    all_csv_path = PIPELINE_DIR / all_csv_name
    train_csv_path = PIPELINE_DIR / train_csv_name
    val_csv_path = PIPELINE_DIR / val_csv_name
    test_csv_path = PIPELINE_DIR / test_csv_name

    if not all_csv_path.exists():
        raise FileNotFoundError(
            "No existe allData.csv.\n"
            "Ejecuta primero:\n"
            "python .\\aplanar_osdb_prototipo_V2.py\n\n"
            f"Ruta esperada:\n{all_csv_path}"
        )

    # Comprobar que allData.csv no esté vacío o incompleto.
    all_df = pd.read_csv(
        all_csv_path,
        low_memory=False,
    )

    if all_df.empty:
        raise RuntimeError(
            "allData.csv está vacío."
        )

    required_all = {
        "eventId",
        "userId",
        "type",
        "typeStr",
    }

    missing_all = required_all.difference(
        all_df.columns
    )

    if missing_all:
        raise RuntimeError(
            "allData.csv parece incompleto. "
            "Faltan columnas: "
            + ", ".join(sorted(missing_all))
        )

    all_df["_label"] = all_df["type"].map(
        normalize_binary_label
    )

    n_events_all = int(
        all_df["eventId"]
        .astype(str)
        .nunique()
    )

    positive_events_all = int(
        all_df.loc[
            all_df["_label"] == 1,
            "eventId",
        ]
        .astype(str)
        .nunique()
    )

    negative_events_all = int(
        all_df.loc[
            all_df["_label"] == 0,
            "eventId",
        ]
        .astype(str)
        .nunique()
    )

    print("")
    print("CONTROL PREVIO DE allData.csv")
    print("-" * 72)
    print(f"Filas: {len(all_df)}")
    print(f"Eventos únicos: {n_events_all}")
    print(
        "Eventos positivos: "
        f"{positive_events_all}"
    )
    print(
        "Eventos negativos: "
        f"{negative_events_all}"
    )

    if positive_events_all < 3:
        raise RuntimeError(
            "Hay menos de 3 eventos positivos; "
            "no se puede hacer una división estratificada "
            "train/val/test fiable."
        )

    if negative_events_all < 3:
        raise RuntimeError(
            "Hay menos de 3 eventos negativos; "
            "no se puede hacer una división estratificada "
            "train/val/test fiable."
        )

    # --------------------------------------------------------
    # Eliminar divisiones anteriores si se desea regenerar.
    # --------------------------------------------------------
    if REGENERAR_DIVISION:
        for path in [
            train_csv_path,
            val_csv_path,
            test_csv_path,
        ]:
            if path.exists():
                path.unlink()

    # --------------------------------------------------------
    # Ejecutar la función oficial de nnTraining2.
    # kFold=1 y nestedKfold=1 producen una sola división.
    # --------------------------------------------------------
    print("")
    print("DIVIDIENDO POR eventId...")
    print("-" * 72)

    splitData.splitData(
        configObj=config,
        kFold=1,
        nestedKfold=1,
        outDir=str(PIPELINE_DIR),
        debug=True,
    )

    # --------------------------------------------------------
    # Cargar resultados.
    # --------------------------------------------------------
    train_df = load_split_csv(train_csv_path)
    val_df = load_split_csv(val_csv_path)
    test_df = load_split_csv(test_csv_path)

    train_events = unique_event_table(
        train_df,
        "train",
    )

    val_events = unique_event_table(
        val_df,
        "validation",
    )

    test_events = unique_event_table(
        test_df,
        "test",
    )

    # --------------------------------------------------------
    # Verificar fuga de eventos.
    # --------------------------------------------------------
    overlap_train_val = overlap(
        train_events["eventId"],
        val_events["eventId"],
    )

    overlap_train_test = overlap(
        train_events["eventId"],
        test_events["eventId"],
    )

    overlap_val_test = overlap(
        val_events["eventId"],
        test_events["eventId"],
    )

    if (
        overlap_train_val
        or overlap_train_test
        or overlap_val_test
    ):
        raise RuntimeError(
            "ERROR: existe fuga de eventId entre conjuntos.\n"
            f"train-val: {sorted(overlap_train_val)}\n"
            f"train-test: {sorted(overlap_train_test)}\n"
            f"val-test: {sorted(overlap_val_test)}"
        )

    # --------------------------------------------------------
    # Verificar que la unión recupera todos los eventos.
    # --------------------------------------------------------
    all_events_set = set_as_strings(
        all_df["eventId"]
    )

    split_events_set = (
        set_as_strings(train_events["eventId"])
        | set_as_strings(val_events["eventId"])
        | set_as_strings(test_events["eventId"])
    )

    missing_events = (
        all_events_set - split_events_set
    )

    extra_events = (
        split_events_set - all_events_set
    )

    if missing_events or extra_events:
        raise RuntimeError(
            "La unión de los conjuntos no coincide "
            "con allData.csv.\n"
            f"Faltantes: {sorted(missing_events)}\n"
            f"Extras: {sorted(extra_events)}"
        )

    # --------------------------------------------------------
    # Reportar superposición de usuarios, sin considerarla error.
    # --------------------------------------------------------
    users_train_val = overlap(
        train_df["userId"],
        val_df["userId"],
    )

    users_train_test = overlap(
        train_df["userId"],
        test_df["userId"],
    )

    users_val_test = overlap(
        val_df["userId"],
        test_df["userId"],
    )

    # --------------------------------------------------------
    # Métricas.
    # --------------------------------------------------------
    metrics = {
        "train": split_metrics(
            train_df,
            train_events,
        ),
        "validation": split_metrics(
            val_df,
            val_events,
        ),
        "test": split_metrics(
            test_df,
            test_events,
        ),
    }

    manifest = pd.concat(
        [
            train_events,
            val_events,
            test_events,
        ],
        ignore_index=True,
    )

    manifest_path = (
        PIPELINE_DIR /
        "manifiesto_division_eventos.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    summary_path = (
        PIPELINE_DIR /
        "resumen_division_eventos.txt"
    )

    test_prop = float(
        config
        .get("dataProcessing", {})
        .get("testProp", 0.0)
    )

    validation_prop = float(
        config
        .get("dataProcessing", {})
        .get("validationProp", 0.0)
    )

    train_prop = (
        1.0 -
        test_prop -
        validation_prop
    )

    lines = [
        "DIVISIÓN OSDB POR EVENTO",
        "=" * 72,
        "",
        f"Entrada: {all_csv_path}",
        f"Semilla: {config.get('randomSeed')}",
        "",
        "PROPORCIONES SOLICITADAS",
        f"Train: {train_prop:.3f}",
        f"Validation: {validation_prop:.3f}",
        f"Test: {test_prop:.3f}",
        "",
        "RESULTADOS",
    ]

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        item = metrics[split_name]

        lines.extend(
            [
                "",
                split_name.upper(),
                f"Filas: {item['rows']}",
                f"Eventos: {item['events']}",
                f"Eventos positivos: "
                f"{item['positive_events']}",
                f"Eventos negativos: "
                f"{item['negative_events']}",
                f"Filas positivas: "
                f"{item['positive_rows']}",
                f"Filas negativas: "
                f"{item['negative_rows']}",
                f"Usuarios: {item['users']}",
                f"Filas completas M000..M124: "
                f"{item['complete_magnitude_rows']}",
            ]
        )

    lines.extend(
        [
            "",
            "CONTROL DE FUGA DE EVENTOS",
            f"Train ∩ Validation: "
            f"{len(overlap_train_val)}",
            f"Train ∩ Test: "
            f"{len(overlap_train_test)}",
            f"Validation ∩ Test: "
            f"{len(overlap_val_test)}",
            "",
            "SUPERPOSICIÓN DE USUARIOS",
            "(Informativa: esta división es por evento, "
            "no por usuario.)",
            f"Train ∩ Validation: "
            f"{len(users_train_val)}",
            f"Train ∩ Test: "
            f"{len(users_train_test)}",
            f"Validation ∩ Test: "
            f"{len(users_val_test)}",
            "",
            "VEREDICTO",
            "Fuga de eventId: NO",
            f"Eventos recuperados: "
            f"{len(split_events_set)}/{len(all_events_set)}",
        ]
    )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("")
    print("=" * 72)
    print("DIVISIÓN COMPLETADA")
    print("=" * 72)

    for split_name, item in metrics.items():
        print(
            f"{split_name}: "
            f"{item['events']} eventos, "
            f"{item['rows']} filas, "
            f"{item['positive_events']} positivos, "
            f"{item['negative_events']} negativos"
        )

    print("")
    print("Fuga de eventos: 0")
    print(
        "Nota: puede haber usuarios repetidos entre "
        "conjuntos porque la división es por evento."
    )
    print(f"Manifiesto: {manifest_path}")
    print(f"Resumen: {summary_path}")


if __name__ == "__main__":
    main()

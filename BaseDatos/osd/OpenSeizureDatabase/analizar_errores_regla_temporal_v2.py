"""
analizar_errores_regla_temporal_v2.py
=====================================

Analiza los errores de la regla temporal V2 usando SOLO VALIDACIÓN.

Entradas:
- probabilidades_validation.csv
- detalle_eventos_mejor_regla.csv
- reglas_candidatas_validacion.json
- manifiesto_subconjunto_v2.csv

Salidas:
- eventos_validacion_clasificados.csv
- falsos_positivos.csv
- falsos_negativos.csv
- resumen_por_categoria_negativa.csv
- resumen_por_subtipo_negativo.csv
- comparacion_features_fp_tn.csv
- comparacion_features_fn_tp.csv
- resumen_analisis_errores.txt
- gráficas de probabilidades por evento

NO usa TEST.
NO cambia el modelo.
NO optimiza nuevos umbrales.

Ejecución:
    python .\\analizar_errores_regla_temporal_v2.py
"""

from __future__ import annotations

import json
import os
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# RUTAS
# ============================================================

BASELINE_DIR = Path(
    r"C:\Users\cesar\osd\pipeline_prototipo_v2"
    r"\baseline_fft_logreg_v2"
)

RULE_DIR = BASELINE_DIR / "regla_temporal_validacion"

WINDOW_PROB_PATH = (
    BASELINE_DIR / "probabilidades_validation.csv"
)

EVENT_DETAIL_PATH = (
    RULE_DIR / "detalle_eventos_mejor_regla.csv"
)

RULE_JSON_PATH = (
    RULE_DIR / "reglas_candidatas_validacion.json"
)

MANIFEST_PATH = Path(
    r"C:\Users\cesar\osd\osdb_subset_v2"
    r"\manifiesto_subconjunto_v2.csv"
)

OUTPUT_DIR = RULE_DIR / "analisis_errores"


# ============================================================
# UTILIDADES
# ============================================================

def normalize_event_id(value: Any) -> str:
    """
    Normaliza eventId para poder unir CSV que pueden leer el identificador
    como int, float o texto.
    """
    if pd.isna(value):
        return "<VACIO>"

    text = str(value).strip()

    if text.endswith(".0"):
        candidate = text[:-2]

        if candidate.replace("-", "").isdigit():
            text = candidate

    return text


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    normalized = {
        str(column).strip().lower(): str(column)
        for column in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    if required:
        raise RuntimeError(
            "No se encontró ninguna columna entre: "
            + ", ".join(candidates)
        )

    return None


def load_selected_rule(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    candidate = data.get("selected_candidate")

    if not isinstance(candidate, dict):
        raise RuntimeError(
            "No se encontró selected_candidate en "
            "reglas_candidatas_validacion.json."
        )

    return candidate


def classify_outcome(
    true_label: int,
    predicted_label: int,
) -> str:
    if true_label == 1 and predicted_label == 1:
        return "TP"

    if true_label == 0 and predicted_label == 0:
        return "TN"

    if true_label == 0 and predicted_label == 1:
        return "FP"

    if true_label == 1 and predicted_label == 0:
        return "FN"

    return "INVALID"


def feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    metadata = {
        "source_row",
        "eventId",
        "event_id_norm",
        "userId",
        "typeStr",
        "label",
        "probability_seizure",
        "window_index",
        "window_positive",
    }

    output: list[str] = []

    for column in df.columns:
        if column in metadata:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            output.append(column)

    return output


def summarize_feature_difference(
    windows: pd.DataFrame,
    group_a: str,
    group_b: str,
    event_outcomes: pd.DataFrame,
) -> pd.DataFrame:
    mapping = dict(
        zip(
            event_outcomes["event_id_norm"],
            event_outcomes["outcome"],
        )
    )

    work = windows.copy()
    work["outcome"] = work["event_id_norm"].map(mapping)

    cols = feature_columns(work)
    rows: list[dict[str, Any]] = []

    for column in cols:
        a = pd.to_numeric(
            work.loc[work["outcome"] == group_a, column],
            errors="coerce",
        ).dropna()

        b = pd.to_numeric(
            work.loc[work["outcome"] == group_b, column],
            errors="coerce",
        ).dropna()

        if a.empty or b.empty:
            continue

        mean_a = float(a.mean())
        mean_b = float(b.mean())

        pooled_scale = float(
            np.sqrt(
                (
                    a.var(ddof=1)
                    + b.var(ddof=1)
                )
                / 2.0
            )
        )

        standardized_difference = (
            (mean_a - mean_b) / pooled_scale
            if pooled_scale > 0
            else 0.0
        )

        rows.append(
            {
                "feature": column,
                f"mean_{group_a}": mean_a,
                f"mean_{group_b}": mean_b,
                "difference": mean_a - mean_b,
                "standardized_difference": standardized_difference,
                f"n_windows_{group_a}": len(a),
                f"n_windows_{group_b}": len(b),
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result["absolute_standardized_difference"] = (
            result["standardized_difference"].abs()
        )

        result = result.sort_values(
            by="absolute_standardized_difference",
            ascending=False,
        )

    return result


def save_event_probability_plot(
    event_id: str,
    outcome: str,
    window_rows: pd.DataFrame,
    threshold: float,
    required_consecutive: int,
    output_path: Path,
    subtype: str,
    category: str,
) -> None:
    ordered = window_rows.sort_values(
        by="window_index"
    )

    x = ordered["window_index"].to_numpy()
    y = ordered["probability_seizure"].to_numpy()

    plt.figure(figsize=(9, 4.5))
    plt.plot(x, y, marker="o", linewidth=1.4)
    plt.axhline(
        threshold,
        linestyle="--",
        label=f"Umbral = {threshold:.4f}",
    )
    plt.ylim(0.0, 1.0)
    plt.xlabel("Ventana consecutiva de 5 s")
    plt.ylabel("Probabilidad de crisis")
    plt.title(
        f"Evento {event_id} — {outcome}\n"
        f"{category} | {subtype} | "
        f"k={required_consecutive}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        WINDOW_PROB_PATH,
        EVENT_DETAIL_PATH,
        RULE_JSON_PATH,
        MANIFEST_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró:\n{path}"
            )

    selected_rule = load_selected_rule(
        RULE_JSON_PATH
    )

    threshold = float(
        selected_rule["threshold"]
    )

    required_consecutive = int(
        round(
            float(
                selected_rule["required_consecutive"]
            )
        )
    )

    windows = pd.read_csv(
        WINDOW_PROB_PATH,
        low_memory=False,
    )

    details = pd.read_csv(
        EVENT_DETAIL_PATH,
        low_memory=False,
    )

    manifest = pd.read_csv(
        MANIFEST_PATH,
        low_memory=False,
    )

    # --------------------------------------------------------
    # Detectar columnas.
    # --------------------------------------------------------
    window_event_col = find_column(
        windows,
        ["eventId", "event_id"],
    )

    window_probability_col = find_column(
        windows,
        [
            "probability_seizure",
            "probability",
            "prob",
        ],
    )

    detail_event_col = find_column(
        details,
        ["eventId", "event_id"],
    )

    detail_true_col = find_column(
        details,
        ["true_label", "label"],
    )

    detail_pred_col = find_column(
        details,
        ["predicted_label", "prediction"],
    )

    manifest_event_col = find_column(
        manifest,
        ["event_id", "eventId"],
    )

    manifest_category_col = find_column(
        manifest,
        ["selection_category"],
        required=False,
    )

    manifest_subtype_col = find_column(
        manifest,
        ["subtype", "subType"],
        required=False,
    )

    manifest_source_col = find_column(
        manifest,
        ["data_source", "dataSourceName"],
        required=False,
    )

    # --------------------------------------------------------
    # Normalizar identificadores.
    # --------------------------------------------------------
    windows = windows.copy()
    details = details.copy()
    manifest = manifest.copy()

    windows["event_id_norm"] = (
        windows[window_event_col]
        .map(normalize_event_id)
    )

    details["event_id_norm"] = (
        details[detail_event_col]
        .map(normalize_event_id)
    )

    manifest["event_id_norm"] = (
        manifest[manifest_event_col]
        .map(normalize_event_id)
    )

    windows = windows.rename(
        columns={
            window_probability_col:
            "probability_seizure"
        }
    )

    windows["probability_seizure"] = pd.to_numeric(
        windows["probability_seizure"],
        errors="raise",
    )

    windows["window_index"] = (
        windows.groupby(
            "event_id_norm",
            sort=False,
        )
        .cumcount()
    )

    windows["window_positive"] = (
        windows["probability_seizure"]
        >= threshold
    ).astype(int)

    details["true_label"] = pd.to_numeric(
        details[detail_true_col],
        errors="raise",
    ).astype(int)

    details["predicted_label"] = pd.to_numeric(
        details[detail_pred_col],
        errors="raise",
    ).astype(int)

    details["outcome"] = [
        classify_outcome(true, pred)
        for true, pred in zip(
            details["true_label"],
            details["predicted_label"],
        )
    ]

    # --------------------------------------------------------
    # Preparar metadatos del manifiesto.
    # --------------------------------------------------------
    manifest_keep = ["event_id_norm"]

    rename_manifest: dict[str, str] = {}

    if manifest_category_col:
        manifest_keep.append(
            manifest_category_col
        )
        rename_manifest[
            manifest_category_col
        ] = "selection_category"

    if manifest_subtype_col:
        manifest_keep.append(
            manifest_subtype_col
        )
        rename_manifest[
            manifest_subtype_col
        ] = "subtype"

    if manifest_source_col:
        manifest_keep.append(
            manifest_source_col
        )
        rename_manifest[
            manifest_source_col
        ] = "data_source"

    manifest_unique = (
        manifest[manifest_keep]
        .drop_duplicates(
            subset=["event_id_norm"]
        )
        .rename(columns=rename_manifest)
    )

    events = details.merge(
        manifest_unique,
        on="event_id_norm",
        how="left",
        validate="one_to_one",
    )

    for column in [
        "selection_category",
        "subtype",
        "data_source",
    ]:
        if column not in events.columns:
            events[column] = "<NO_DISPONIBLE>"

        events[column] = (
            events[column]
            .fillna("<VACÍO>")
            .astype(str)
        )

    # --------------------------------------------------------
    # Añadir estadísticas directas de ventanas.
    # --------------------------------------------------------
    window_stats = (
        windows.groupby(
            "event_id_norm",
            sort=False,
        )
        .agg(
            probability_mean=(
                "probability_seizure",
                "mean",
            ),
            probability_max=(
                "probability_seizure",
                "max",
            ),
            probability_q70=(
                "probability_seizure",
                lambda values: float(
                    np.quantile(values, 0.70)
                ),
            ),
            n_windows=(
                "probability_seizure",
                "size",
            ),
            positive_windows=(
                "window_positive",
                "sum",
            ),
        )
        .reset_index()
    )

    events = events.merge(
        window_stats,
        on="event_id_norm",
        how="left",
        suffixes=("", "_recomputed"),
    )

    # --------------------------------------------------------
    # Separar resultados.
    # --------------------------------------------------------
    false_positives = events.loc[
        events["outcome"] == "FP"
    ].copy()

    false_negatives = events.loc[
        events["outcome"] == "FN"
    ].copy()

    true_positives = events.loc[
        events["outcome"] == "TP"
    ].copy()

    true_negatives = events.loc[
        events["outcome"] == "TN"
    ].copy()

    events_path = (
        OUTPUT_DIR
        / "eventos_validacion_clasificados.csv"
    )

    fp_path = (
        OUTPUT_DIR
        / "falsos_positivos.csv"
    )

    fn_path = (
        OUTPUT_DIR
        / "falsos_negativos.csv"
    )

    events.to_csv(events_path, index=False)
    false_positives.to_csv(fp_path, index=False)
    false_negatives.to_csv(fn_path, index=False)

    # --------------------------------------------------------
    # Resumen de negativos por categoría/subtipo.
    # --------------------------------------------------------
    negative_events = events.loc[
        events["true_label"] == 0
    ].copy()

    category_summary = (
        negative_events.groupby(
            "selection_category",
            dropna=False,
        )
        .agg(
            negative_events=(
                "event_id_norm",
                "size",
            ),
            false_positives=(
                "predicted_label",
                "sum",
            ),
        )
        .reset_index()
    )

    category_summary[
        "false_positive_rate"
    ] = (
        category_summary["false_positives"]
        / category_summary["negative_events"]
    )

    subtype_summary = (
        negative_events.groupby(
            ["selection_category", "subtype"],
            dropna=False,
        )
        .agg(
            negative_events=(
                "event_id_norm",
                "size",
            ),
            false_positives=(
                "predicted_label",
                "sum",
            ),
        )
        .reset_index()
    )

    subtype_summary[
        "false_positive_rate"
    ] = (
        subtype_summary["false_positives"]
        / subtype_summary["negative_events"]
    )

    subtype_summary = subtype_summary.sort_values(
        by=[
            "false_positives",
            "false_positive_rate",
            "negative_events",
        ],
        ascending=[False, False, False],
    )

    category_path = (
        OUTPUT_DIR
        / "resumen_por_categoria_negativa.csv"
    )

    subtype_path = (
        OUTPUT_DIR
        / "resumen_por_subtipo_negativo.csv"
    )

    category_summary.to_csv(
        category_path,
        index=False,
    )

    subtype_summary.to_csv(
        subtype_path,
        index=False,
    )

    # --------------------------------------------------------
    # Comparación de características.
    # --------------------------------------------------------
    fp_tn_features = summarize_feature_difference(
        windows=windows,
        group_a="FP",
        group_b="TN",
        event_outcomes=events,
    )

    fn_tp_features = summarize_feature_difference(
        windows=windows,
        group_a="FN",
        group_b="TP",
        event_outcomes=events,
    )

    fp_tn_path = (
        OUTPUT_DIR
        / "comparacion_features_fp_tn.csv"
    )

    fn_tp_path = (
        OUTPUT_DIR
        / "comparacion_features_fn_tp.csv"
    )

    fp_tn_features.to_csv(
        fp_tn_path,
        index=False,
    )

    fn_tp_features.to_csv(
        fn_tp_path,
        index=False,
    )

    # --------------------------------------------------------
    # Gráficas de todos los FP y FN.
    # --------------------------------------------------------
    plot_dir = OUTPUT_DIR / "graficas_eventos"
    plot_dir.mkdir(exist_ok=True)

    error_events = pd.concat(
        [
            false_positives,
            false_negatives,
        ],
        ignore_index=True,
    )

    for _, event_row in error_events.iterrows():
        event_id = event_row["event_id_norm"]
        outcome = event_row["outcome"]

        event_windows = windows.loc[
            windows["event_id_norm"]
            == event_id
        ]

        safe_id = re_safe_filename(event_id)

        output_path = (
            plot_dir
            / f"{outcome}_evento_{safe_id}.png"
        )

        save_event_probability_plot(
            event_id=event_id,
            outcome=outcome,
            window_rows=event_windows,
            threshold=threshold,
            required_consecutive=required_consecutive,
            output_path=output_path,
            subtype=str(event_row["subtype"]),
            category=str(
                event_row["selection_category"]
            ),
        )

    # --------------------------------------------------------
    # Resumen de texto.
    # --------------------------------------------------------
    summary_path = (
        OUTPUT_DIR
        / "resumen_analisis_errores.txt"
    )

    lines = [
        "ANÁLISIS DE ERRORES — REGLA TEMPORAL V2",
        "=" * 78,
        "",
        "Este análisis usa únicamente VALIDACIÓN.",
        "TEST permanece sin utilizar.",
        "",
        "REGLA ANALIZADA",
        f"Umbral: {threshold}",
        f"Ventanas consecutivas requeridas: "
        f"{required_consecutive}",
        "",
        "MATRIZ DE CONFUSIÓN POR EVENTO",
        f"TP: {len(true_positives)}",
        f"TN: {len(true_negatives)}",
        f"FP: {len(false_positives)}",
        f"FN: {len(false_negatives)}",
        "",
        "FALSOS POSITIVOS",
    ]

    if false_positives.empty:
        lines.append("Ninguno.")
    else:
        for _, row in false_positives.iterrows():
            lines.append(
                f"- eventId={row['event_id_norm']} | "
                f"categoría={row['selection_category']} | "
                f"subtipo={row['subtype']} | "
                f"fuente={row['data_source']} | "
                f"maxP={row['probability_max']:.4f} | "
                f"ventanas+={int(row['positive_windows'])}/"
                f"{int(row['n_windows'])}"
            )

    lines.extend(
        [
            "",
            "FALSOS NEGATIVOS",
        ]
    )

    if false_negatives.empty:
        lines.append("Ninguno.")
    else:
        for _, row in false_negatives.iterrows():
            lines.append(
                f"- eventId={row['event_id_norm']} | "
                f"subtipo={row['subtype']} | "
                f"fuente={row['data_source']} | "
                f"maxP={row['probability_max']:.4f} | "
                f"ventanas+={int(row['positive_windows'])}/"
                f"{int(row['n_windows'])}"
            )

    lines.extend(
        [
            "",
            "FPR POR CATEGORÍA NEGATIVA",
            category_summary.to_string(index=False),
            "",
            "SUBTIPOS NEGATIVOS CON MÁS FALSOS POSITIVOS",
            subtype_summary.head(15).to_string(index=False),
            "",
            "CARACTERÍSTICAS QUE MÁS DIFERENCIAN FP DE TN",
            (
                fp_tn_features.head(10).to_string(index=False)
                if not fp_tn_features.empty
                else "No disponible."
            ),
            "",
            "CARACTERÍSTICAS QUE MÁS DIFERENCIAN FN DE TP",
            (
                fn_tp_features.head(10).to_string(index=False)
                if not fn_tp_features.empty
                else "No disponible."
            ),
            "",
            "INTERPRETACIÓN",
            (
                "El objetivo es identificar si los falsos positivos "
                "se concentran en una actividad concreta o si son "
                "heterogéneos. Con ese resultado se decidirá entre:"
            ),
            "- compuerta de intensidad de movimiento;",
            "- nuevas características de ritmicidad;",
            "- incorporar rawData3D;",
            "- conservar dos ventanas como WARNING y exigir otra "
            "condición para ALARM.",
        ]
    )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("")
    print("=" * 78)
    print("ANÁLISIS DE ERRORES COMPLETADO")
    print("=" * 78)
    print(f"TP={len(true_positives)}")
    print(f"TN={len(true_negatives)}")
    print(f"FP={len(false_positives)}")
    print(f"FN={len(false_negatives)}")
    print(f"Resumen: {summary_path}")
    print(f"Falsos positivos: {fp_path}")
    print(f"Falsos negativos: {fn_path}")
    print("")
    print("TEST todavía no fue utilizado.")


def re_safe_filename(value: str) -> str:
    output = []

    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            output.append(char)
        else:
            output.append("_")

    return "".join(output)


if __name__ == "__main__":
    main()

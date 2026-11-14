"""
extraer_features_verificar_buffer.py
====================================

Siguiente tramo del prototipo OSDB:

1. Lee:
   - trainData.csv
   - valData.csv
   - testData.csv

2. Ejecuta extractFeatures oficial con:
   window = 125
   step   = 125
   simpleMagnitudeOnly = true

3. Genera:
   - trainFeatures.csv
   - valFeatures.csv
   - testFeatures.csv

4. Verifica dinÃ¡micamente la geometrÃ­a real de DeepEpi:
   - Cada fila de features contiene 125 muestras (5 s).
   - El modelo mantiene un buffer de 750 muestras (30 s).
   - Detecta si resetAccBuf() deja el buffer vacÃ­o o prellenado.
   - Calcula cuÃ¡ntos datapoints iniciales son de calentamiento.
   - No presupone cÃ³mo se comporta la versiÃ³n instalada del repositorio.

5. Llama a nnTrainer.df2trainingData para confirmar:
   - cantidad real de vectores;
   - longitud real de cada vector;
   - clases producidas.

NO entrena el modelo.

EjecuciÃ³n desde:
C:\\Users\\cesar\\osd\\OpenSeizureDatabase

Comando:
python .\\extraer_features_verificar_buffer.py
"""

from __future__ import annotations

import gc
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from user_tools.nnTraining2 import extractFeatures
from user_tools.nnTraining2 import nnTrainer


# ============================================================
# RUTAS
# ============================================================

REPO_DIR = Path(r"C:\Users\cesar\osd\OpenSeizureDatabase")
CONFIG_PATH = REPO_DIR / "nnConfig_prototipo_pytorch.json"
PIPELINE_DIR = Path(r"C:\Users\cesar\osd\pipeline_prototipo")

REGENERAR_FEATURES = False


# ============================================================
# FUNCIONES
# ============================================================

def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontrÃ³ la configuraciÃ³n:\n{path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def import_class(class_path: str):
    """
    Importa:
      user_tools.nnTraining2.deepEpiCnnModel_torch.DeepEpiCnnModelPyTorch
    """
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def collect_magnitude_columns(columns: list[str]) -> list[str]:
    """
    Admite M000...M124 o M000_t-0...M124_t-0.
    """
    with_suffix = [
        c
        for c in columns
        if (
            isinstance(c, str)
            and c.startswith("M")
            and c.endswith("_t-0")
            and c[1:-4].isdigit()
        )
    ]

    if with_suffix:
        return sorted(
            with_suffix,
            key=lambda c: int(c[1:-4]),
        )

    plain = [
        c
        for c in columns
        if (
            isinstance(c, str)
            and len(c) == 4
            and c.startswith("M")
            and c[1:].isdigit()
        )
    ]

    return sorted(
        plain,
        key=lambda c: int(c[1:]),
    )


def normalise_label(value: Any) -> int:
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

    number = float(value)

    if number == 1:
        return 1

    if number == 0:
        return 0

    raise ValueError(f"Etiqueta no binaria: {value!r}")


def event_manifest(
    df: pd.DataFrame,
    split_name: str,
    blocks_per_buffer: int,
    reset_prefill_samples: int,
) -> pd.DataFrame:
    """
    Una fila por evento con su etiqueta, nÃºmero de bloques y geometrÃ­a
    esperada segÃºn el comportamiento real de resetAccBuf().
    """
    work = df.copy()
    work["_label"] = work["type"].map(normalise_label)

    grouped = (
        work.groupby("eventId", sort=False, dropna=False)
        .agg(
            userId=("userId", "first"),
            typeStr=("typeStr", "first"),
            label=("_label", "first"),
            blocks_5s=("eventId", "size"),
        )
        .reset_index()
    )

    if reset_prefill_samples >= blocks_per_buffer * 125:
        # El buffer ya estÃ¡ lleno al comenzar cada evento.
        grouped["expected_output_vectors"] = grouped["blocks_5s"]
    else:
        # Buffer vacÃ­o: se necesitan blocks_per_buffer filas para el primer vector.
        grouped["expected_output_vectors"] = (
            grouped["blocks_5s"] - blocks_per_buffer + 1
        ).clip(lower=0)

    grouped["warmup_blocks"] = grouped["blocks_5s"].clip(
        upper=max(0, blocks_per_buffer - 1)
    )

    grouped["fully_real_30s_vectors"] = (
        grouped["blocks_5s"] - blocks_per_buffer + 1
    ).clip(lower=0)

    grouped["has_fully_real_30s_vector"] = (
        grouped["blocks_5s"] >= blocks_per_buffer
    )

    grouped.insert(0, "split", split_name)
    return grouped


def inspect_feature_file(
    feature_path: Path,
    split_name: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    if not feature_path.exists():
        raise FileNotFoundError(
            f"No se generÃ³:\n{feature_path}"
        )

    df = pd.read_csv(
        feature_path,
        low_memory=False,
    )

    required = {
        "eventId",
        "userId",
        "typeStr",
        "type",
    }

    missing = required.difference(df.columns)

    if missing:
        raise RuntimeError(
            f"{feature_path.name} carece de columnas: "
            + ", ".join(sorted(missing))
        )

    magnitude_columns = collect_magnitude_columns(
        list(df.columns)
    )

    if len(magnitude_columns) != 125:
        raise RuntimeError(
            f"{feature_path.name}: se esperaban 125 columnas M; "
            f"se encontraron {len(magnitude_columns)}."
        )

    if df[magnitude_columns].isna().any(axis=None):
        rows_with_missing = int(
            df[magnitude_columns]
            .isna()
            .any(axis=1)
            .sum()
        )
        raise RuntimeError(
            f"{feature_path.name}: {rows_with_missing} filas "
            "tienen muestras M faltantes."
        )

    sample_freq = int(
        config["modelConfig"]["sampleFreq"]
    )

    buffer_seconds = int(
        config["modelConfig"]["bufferSeconds"]
    )

    buffer_samples = sample_freq * buffer_seconds

    if buffer_samples % len(magnitude_columns) != 0:
        raise RuntimeError(
            "bufferSamples no es mÃºltiplo de las 125 muestras "
            "por bloque."
        )

    blocks_per_buffer = (
        buffer_samples // len(magnitude_columns)
    )

    # --------------------------------------------------------
    # Instanciar el wrapper oficial y detectar el comportamiento
    # real de resetAccBuf() en esta versiÃ³n del repositorio.
    # --------------------------------------------------------
    model_class_path = config[
        "modelConfig"
    ]["modelClass"]

    model_class = import_class(
        model_class_path
    )

    probe_model = model_class(
        configObj=config["modelConfig"],
        debug=False,
    )

    probe_model.resetAccBuf()

    reset_prefill_samples = len(
        getattr(probe_model, "accBuf", [])
    )

    warmup_blocks_reported = int(
        probe_model.get_warmup_datapoints(
            samples_per_datapoint=125
        )
    )

    manifest = event_manifest(
        df=df,
        split_name=split_name,
        blocks_per_buffer=blocks_per_buffer,
        reset_prefill_samples=reset_prefill_samples,
    )

    expected_vectors = int(
        manifest["expected_output_vectors"].sum()
    )

    positive_expected = int(
        manifest.loc[
            manifest["label"] == 1,
            "expected_output_vectors",
        ].sum()
    )

    negative_expected = int(
        manifest.loc[
            manifest["label"] == 0,
            "expected_output_vectors",
        ].sum()
    )

    short_events = int(
        (~manifest["has_fully_real_30s_vector"]).sum()
    )

    expected_fully_real_vectors = int(
        manifest["fully_real_30s_vectors"].sum()
    )

    # --------------------------------------------------------
    # VerificaciÃ³n real usando el conversor oficial.
    # --------------------------------------------------------
    model_wrapper = model_class(
        configObj=config["modelConfig"],
        debug=False,
    )

    x_list, y_list = nnTrainer.df2trainingData(
        df,
        model_wrapper,
        debug=False,
    )

    actual_vectors = len(x_list)

    if actual_vectors:
        lengths = sorted(
            {
                int(np.asarray(vector).size)
                for vector in x_list
            }
        )
    else:
        lengths = []

    actual_positive = int(
        sum(
            normalise_label(label) == 1
            for label in y_list
        )
    )

    actual_negative = int(
        sum(
            normalise_label(label) == 0
            for label in y_list
        )
    )

    if actual_vectors != expected_vectors:
        raise RuntimeError(
            f"{split_name}: el cÃ¡lculo esperado dio "
            f"{expected_vectors}, pero df2trainingData produjo "
            f"{actual_vectors}."
        )

    if lengths != [buffer_samples]:
        raise RuntimeError(
            f"{split_name}: longitudes inesperadas: {lengths}; "
            f"se esperaba [{buffer_samples}]."
        )

    if (
        actual_positive != positive_expected
        or actual_negative != negative_expected
    ):
        raise RuntimeError(
            f"{split_name}: las clases reales no coinciden con "
            "el cÃ¡lculo por evento."
        )

    metrics = {
        "split": split_name,
        "feature_rows_5s": len(df),
        "events": len(manifest),
        "magnitude_columns": len(magnitude_columns),
        "sample_freq_hz": sample_freq,
        "buffer_seconds": buffer_seconds,
        "buffer_samples": buffer_samples,
        "blocks_per_buffer": blocks_per_buffer,
        "reset_prefill_samples": reset_prefill_samples,
        "warmup_blocks_reported": warmup_blocks_reported,
        "events_without_fully_real_30s": short_events,
        "expected_output_vectors": expected_vectors,
        "actual_output_vectors": actual_vectors,
        "fully_real_30s_vectors_possible": expected_fully_real_vectors,
        "positive_output_vectors": actual_positive,
        "negative_output_vectors": actual_negative,
        "vector_lengths": repr(lengths),
    }

    del x_list, y_list, df, model_wrapper
    gc.collect()

    return metrics, manifest


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:
    config = load_config(CONFIG_PATH)

    data_names = config["dataFileNames"]

    jobs = [
        {
            "split": "train",
            "input": PIPELINE_DIR
            / data_names["trainDataFileCsv"],
            "output": PIPELINE_DIR
            / data_names["trainFeaturesFileCsv"],
        },
        {
            "split": "validation",
            "input": PIPELINE_DIR
            / data_names["valDataFileCsv"],
            "output": PIPELINE_DIR
            / data_names["valFeaturesFileCsv"],
        },
        {
            "split": "test",
            "input": PIPELINE_DIR
            / data_names["testDataFileCsv"],
            "output": PIPELINE_DIR
            / data_names["testFeaturesFileCsv"],
        },
    ]

    for job in jobs:
        if not job["input"].exists():
            raise FileNotFoundError(
                f"No existe {job['input']}.\n"
                "Ejecuta antes dividir_osdb_prototipo.py."
            )

    print("")
    print("CONFIGURACIÃ“N DE GEOMETRÃA")
    print("-" * 72)
    print(
        "Bloque OSDB: "
        f"{config['dataProcessing']['window']} muestras"
    )
    print(
        "Paso de extracciÃ³n: "
        f"{config['dataProcessing']['step']} muestras"
    )
    print(
        "Frecuencia: "
        f"{config['modelConfig']['sampleFreq']} Hz"
    )
    print(
        "BÃºfer DeepEpi: "
        f"{config['modelConfig']['bufferSeconds']} s"
    )

    # --------------------------------------------------------
    # Extraer features de cada conjunto.
    # --------------------------------------------------------
    for job in jobs:
        output_path: Path = job["output"]

        if REGENERAR_FEATURES and output_path.exists():
            output_path.unlink()

        temporary_path = Path(
            str(output_path) + ".tmp.csv"
        )

        if temporary_path.exists():
            temporary_path.unlink()

        if not output_path.exists():
            print("")
            print("=" * 72)
            print(
                f"EXTRAYENDO FEATURES: "
                f"{job['split'].upper()}"
            )
            print("=" * 72)
            print(f"Entrada: {job['input']}")
            print(f"Salida: {output_path}")

            # Evitar que streamTmpOut reutilizado apunte a otro split.
            config.setdefault(
                "dataFileNames",
                {},
            )
            config["dataFileNames"][
                "streamTmpOut"
            ] = str(temporary_path)

            extractFeatures.extractFeatures(
                inFname=str(job["input"]),
                outFname=str(output_path),
                configObj=config,
                debug=False,
            )

    # --------------------------------------------------------
    # VerificaciÃ³n de 125 -> 750.
    # --------------------------------------------------------
    all_metrics: list[dict[str, Any]] = []
    manifests: list[pd.DataFrame] = []

    for job in jobs:
        print("")
        print(
            f"VERIFICANDO BUFFER: "
            f"{job['split'].upper()}"
        )

        metrics, manifest = inspect_feature_file(
            feature_path=job["output"],
            split_name=job["split"],
            config=config,
        )

        all_metrics.append(metrics)
        manifests.append(manifest)

        print(
            f"  Filas de 5 s: "
            f"{metrics['feature_rows_5s']}"
        )
        print(
            f"  Vectores producidos: "
            f"{metrics['actual_output_vectors']}"
        )
        print(
            f"  Positivos: "
            f"{metrics['positive_output_vectors']}"
        )
        print(
            f"  Negativos: "
            f"{metrics['negative_output_vectors']}"
        )
        print(
            f"  Longitud verificada: "
            f"{metrics['vector_lengths']}"
        )

    metrics_df = pd.DataFrame(all_metrics)
    manifest_df = pd.concat(
        manifests,
        ignore_index=True,
    )

    metrics_path = (
        PIPELINE_DIR
        / "resumen_geometria_125_a_750.csv"
    )

    manifest_path = (
        PIPELINE_DIR
        / "manifiesto_eventos_buffer_30s.csv"
    )

    text_path = (
        PIPELINE_DIR
        / "resumen_geometria_125_a_750.txt"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    manifest_df.to_csv(
        manifest_path,
        index=False,
    )

    lines = [
        "VERIFICACIÃ“N 125 MUESTRAS -> BÃšFER DE 750",
        "=" * 72,
        "",
        "InterpretaciÃ³n:",
        "- Cada fila de features contiene 125 muestras = 5 s.",
        "- DeepEpi usa bufferSeconds=30 a 25 Hz = 750 muestras.",
        "- Cada vector del modelo contiene los Ãºltimos 6 bloques.",
        "- El script detecta si resetAccBuf prellena o deja vacÃ­o el buffer.",
        "- Los primeros 5 datapoints contienen calentamiento/prefill cuando",
        "  el modelo puntÃºa desde el primer bloque.",
        "",
    ]

    for item in all_metrics:
        lines.extend(
            [
                item["split"].upper(),
                f"Filas de 5 s: "
                f"{item['feature_rows_5s']}",
                f"Eventos: {item['events']}",
                f"Buffer prellenado al resetear: "
                f"{item['reset_prefill_samples']} muestras",
                f"Warm-up reportado: "
                f"{item['warmup_blocks_reported']} bloques",
                f"Eventos sin ventana 30 s completamente real: "
                f"{item['events_without_fully_real_30s']}",
                f"Vectores producidos: "
                f"{item['actual_output_vectors']}",
                f"Vectores 30 s completamente reales posibles: "
                f"{item['fully_real_30s_vectors_possible']}",
                f"Vectores positivos: "
                f"{item['positive_output_vectors']}",
                f"Vectores negativos: "
                f"{item['negative_output_vectors']}",
                f"Longitudes: "
                f"{item['vector_lengths']}",
                "",
            ]
        )

    text_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("")
    print("=" * 72)
    print("EXTRACCIÃ“N Y VERIFICACIÃ“N COMPLETADAS")
    print("=" * 72)
    print(f"Resumen CSV: {metrics_path}")
    print(f"Resumen TXT: {text_path}")
    print(f"Manifiesto: {manifest_path}")
    print("")
    print(
        "TodavÃ­a NO se ha entrenado ningÃºn modelo."
    )


if __name__ == "__main__":
    main()


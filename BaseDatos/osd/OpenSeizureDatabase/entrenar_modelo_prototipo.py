"""
entrenar_modelo_prototipo.py
============================

Primer entrenamiento reducido del detector OSDB con PyTorch.

Qué hace:
1. Comprueba trainFeatures.csv, valFeatures.csv y testFeatures.csv.
2. Copia esos archivos a una carpeta aislada de ejecución.
3. Crea una configuración de entrenamiento "smoke test":
   - PyTorch
   - DeepEpi 1D CNN
   - entrada de 750 muestras = 30 s a 25 Hz
   - 3 épocas
   - lotes balanceados
   - sin usar el conjunto de test durante el entrenamiento
4. Ejecuta nnTrainer.trainModel().
5. Guarda:
   - checkpoint .pt
   - training_history.json
   - epoch_metrics.csv
   - log completo
   - resumen_entrenamiento_inicial.txt

NO realiza todavía la evaluación final sobre testFeatures.csv.

Ejecutar desde:
C:\\Users\\cesar\\osd\\OpenSeizureDatabase

Comando:
python .\\entrenar_modelo_prototipo.py
"""

from __future__ import annotations

# Debe establecerse antes de importar nnTrainer/Matplotlib.
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import contextlib
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from user_tools.nnTraining2 import nnTrainer


# ============================================================
# RUTAS
# ============================================================

REPO_DIR = Path(r"C:\Users\cesar\osd\OpenSeizureDatabase")
BASE_CONFIG_PATH = REPO_DIR / "nnConfig_prototipo_pytorch.json"

SOURCE_DATA_DIR = Path(r"C:\Users\cesar\osd\pipeline_prototipo")
RUN_DIR = SOURCE_DATA_DIR / "entrenamiento_01"

MODEL_NAME = "prototipo_osdb_pytorch_smoke"

# True = elimina la ejecución anterior de entrenamiento_01 y empieza de cero.
REINICIAR_ENTRENAMIENTO = True


# ============================================================
# UTILIDADES
# ============================================================

class Tee:
    """Duplica stdout/stderr en consola y archivo."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró:\n{path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def check_feature_file(
    path: Path,
    split_name: str,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {split_name}:\n{path}\n\n"
            "Ejecuta primero:\n"
            "python .\\extraer_features_verificar_buffer.py"
        )

    df = pd.read_csv(path, low_memory=False)

    required_metadata = {
        "eventId",
        "userId",
        "type",
        "typeStr",
    }

    missing = required_metadata.difference(df.columns)

    if missing:
        raise RuntimeError(
            f"{path.name} no contiene: "
            + ", ".join(sorted(missing))
        )

    magnitude_columns = [
        column
        for column in df.columns
        if (
            isinstance(column, str)
            and column.startswith("M")
            and (
                (
                    len(column) == 4
                    and column[1:].isdigit()
                )
                or (
                    column.endswith("_t-0")
                    and column[1:-4].isdigit()
                )
            )
        )
    ]

    if len(magnitude_columns) != 125:
        raise RuntimeError(
            f"{path.name}: se esperaban 125 columnas M; "
            f"se encontraron {len(magnitude_columns)}."
        )

    incomplete_rows = int(
        df[magnitude_columns]
        .isna()
        .any(axis=1)
        .sum()
    )

    if incomplete_rows:
        raise RuntimeError(
            f"{path.name}: {incomplete_rows} filas "
            "tienen muestras incompletas."
        )

    labels = pd.to_numeric(
        df["type"],
        errors="coerce",
    )

    if labels.isna().any():
        raise RuntimeError(
            f"{path.name}: hay etiquetas no numéricas."
        )

    return {
        "split": split_name,
        "rows_5s": len(df),
        "events": int(
            df["eventId"]
            .astype(str)
            .nunique()
        ),
        "positive_rows": int(
            (labels == 1).sum()
        ),
        "negative_rows": int(
            (labels == 0).sum()
        ),
        "users": int(
            df["userId"]
            .astype(str)
            .nunique()
        ),
    }


def prepare_runtime_config(
    base_config: dict[str, Any],
) -> dict[str, Any]:
    # Copia profunda mediante JSON: la configuración solo contiene
    # tipos serializables.
    config = json.loads(
        json.dumps(base_config)
    )

    config["debug"] = True
    config["randomSeed"] = 42

    data_names = config.setdefault(
        "dataFileNames",
        {},
    )

    data_names["trainFeaturesFileCsv"] = (
        "trainFeatures.csv"
    )
    data_names["valFeaturesFileCsv"] = (
        "valFeatures.csv"
    )
    data_names["testFeaturesFileCsv"] = (
        "testFeatures.csv"
    )

    model = config.setdefault(
        "modelConfig",
        {},
    )

    model["framework"] = "pytorch"
    model["modelFname"] = MODEL_NAME

    # Primer entrenamiento reducido.
    model["epochs"] = 3
    model["batchSize"] = 64
    model["trainingVerbosity"] = 1
    model["earlyStoppingPatience"] = 3

    # Mantener arquitectura y geometría DeepEpi.
    model["sampleFreq"] = 25
    model["bufferSeconds"] = 30
    model["nLayers"] = 14
    model["dims"] = 1

    # Entrenamiento por épocas sencillo.
    model["useLrSchedule"] = False
    model["lrStart"] = 0.0001
    model["lrMin"] = 1e-7
    model["lrFactor"] = 0.1
    model["lrPatience"] = 2

    model["useAdamW"] = True
    model["useBalancedBatches"] = True

    # Para este smoke test se relajan los límites de guardado.
    # No son los criterios del modelo final.
    model["saveBestMinSensitivity"] = 0.0
    model["saveBestMaxFpr"] = 1.0
    model["modelSelectionMetric"] = "f_beta"
    model["fBeta"] = 2.0

    return config


def write_final_summary(
    summary_path: Path,
    runtime_config_path: Path,
    model_path: Path,
    feature_metrics: list[dict[str, Any]],
    start_time: datetime,
    end_time: datetime,
) -> None:
    epoch_metrics_path = RUN_DIR / "epoch_metrics.csv"
    history_path = RUN_DIR / "training_history.json"

    lines = [
        "PRIMER ENTRENAMIENTO PYTORCH — OSDB PROTOTIPO",
        "=" * 78,
        "",
        f"Inicio: {start_time.isoformat(timespec='seconds')}",
        f"Fin: {end_time.isoformat(timespec='seconds')}",
        f"Configuración ejecutada: {runtime_config_path}",
        f"Dispositivo torch: "
        f"{'CUDA' if torch.cuda.is_available() else 'CPU'}",
        f"Versión torch: {torch.__version__}",
        "",
        "DATOS DE ENTRADA",
    ]

    for item in feature_metrics:
        lines.extend(
            [
                f"{item['split']}:",
                f"  filas de 5 s: {item['rows_5s']}",
                f"  eventos: {item['events']}",
                f"  filas positivas: {item['positive_rows']}",
                f"  filas negativas: {item['negative_rows']}",
                f"  usuarios: {item['users']}",
            ]
        )

    lines.extend(
        [
            "",
            "CONFIGURACIÓN DEL MODELO",
            "Framework: PyTorch",
            "Arquitectura: DeepEpi CNN 1D de 14 capas",
            "Entrada: 750 muestras = 30 s a 25 Hz",
            "Épocas solicitadas: 3",
            "Batch size: 64",
            "Lotes balanceados: sí",
            "Conjunto test usado durante entrenamiento: no",
            "",
            "ARCHIVOS",
            f"Checkpoint esperado: {model_path}",
            f"Checkpoint existe: {model_path.exists()}",
        ]
    )

    if model_path.exists():
        lines.append(
            f"Tamaño del checkpoint: "
            f"{model_path.stat().st_size / (1024**2):.3f} MB"
        )

    lines.extend(
        [
            f"Historial existe: {history_path.exists()}",
            f"Métricas por época existen: "
            f"{epoch_metrics_path.exists()}",
        ]
    )

    if epoch_metrics_path.exists():
        metrics_df = pd.read_csv(
            epoch_metrics_path,
            low_memory=False,
        )

        lines.extend(
            [
                "",
                "MÉTRICAS POR ÉPOCA",
                metrics_df.to_string(index=False),
            ]
        )

        saved_rows = metrics_df[
            metrics_df["checkpoint_saved"]
            .astype(str)
            .str.lower()
            .isin({"true", "1"})
        ]

        if not saved_rows.empty:
            last_saved = saved_rows.iloc[-1]

            lines.extend(
                [
                    "",
                    "ÚLTIMO CHECKPOINT GUARDADO",
                    f"Época: {last_saved.get('epoch')}",
                    f"Val loss: {last_saved.get('val_loss')}",
                    f"Val accuracy: "
                    f"{last_saved.get('val_accuracy')}",
                    f"Sensibilidad: "
                    f"{last_saved.get('sensitivity')}",
                    f"FAR/FPR: {last_saved.get('far')}",
                    f"Motivo: "
                    f"{last_saved.get('save_reason')}",
                ]
            )

    lines.extend(
        [
            "",
            "INTERPRETACIÓN",
            "Este entrenamiento es una prueba de funcionamiento.",
            "Tres épocas no permiten concluir rendimiento clínico.",
            "El conjunto de test permanece reservado para el siguiente tramo.",
        ]
    )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:
    base_config = load_json(
        BASE_CONFIG_PATH
    )

    source_files = {
        "train": SOURCE_DATA_DIR / "trainFeatures.csv",
        "validation": SOURCE_DATA_DIR / "valFeatures.csv",
        "test": SOURCE_DATA_DIR / "testFeatures.csv",
    }

    feature_metrics = [
        check_feature_file(path, split_name)
        for split_name, path in source_files.items()
    ]

    if REINICIAR_ENTRENAMIENTO and RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)

    RUN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Copiar features para aislar la ejecución.
    for split_name, source_path in source_files.items():
        destination_name = {
            "train": "trainFeatures.csv",
            "validation": "valFeatures.csv",
            "test": "testFeatures.csv",
        }[split_name]

        shutil.copy2(
            source_path,
            RUN_DIR / destination_name,
        )

    runtime_config = prepare_runtime_config(
        base_config
    )

    runtime_config_path = (
        RUN_DIR /
        "nnConfig_entrenamiento_01.json"
    )

    runtime_config_path.write_text(
        json.dumps(
            runtime_config,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    model_path = (
        RUN_DIR /
        f"{MODEL_NAME}.pt"
    )

    log_path = (
        RUN_DIR /
        "entrenamiento_consola.log"
    )

    summary_path = (
        RUN_DIR /
        "resumen_entrenamiento_inicial.txt"
    )

    start_time = datetime.now()

    print("")
    print("=" * 78)
    print("PRIMER ENTRENAMIENTO REDUCIDO")
    print("=" * 78)
    print(f"Salida: {RUN_DIR}")
    print(
        "Dispositivo: "
        f"{'CUDA/GPU' if torch.cuda.is_available() else 'CPU'}"
    )
    print("Épocas: 3")
    print("El conjunto test NO se usará durante el entrenamiento.")
    print("")

    with log_path.open(
        "w",
        encoding="utf-8",
        buffering=1,
    ) as log_handle:
        tee_out = Tee(
            sys.__stdout__,
            log_handle,
        )
        tee_err = Tee(
            sys.__stderr__,
            log_handle,
        )

        with (
            contextlib.redirect_stdout(tee_out),
            contextlib.redirect_stderr(tee_err),
        ):
            nnTrainer.trainModel(
                configObj=runtime_config,
                dataDir=str(RUN_DIR),
                debug=True,
            )

    end_time = datetime.now()

    write_final_summary(
        summary_path=summary_path,
        runtime_config_path=runtime_config_path,
        model_path=model_path,
        feature_metrics=feature_metrics,
        start_time=start_time,
        end_time=end_time,
    )

    print("")
    print("=" * 78)
    print("ENTRENAMIENTO INICIAL FINALIZADO")
    print("=" * 78)
    print(f"Modelo: {model_path}")
    print(f"Modelo creado: {model_path.exists()}")
    print(f"Log: {log_path}")
    print(f"Resumen: {summary_path}")
    print("")
    print(
        "Una advertencia sobre conversión .pte/ExecuTorch "
        "puede aparecer al final y no invalida el checkpoint .pt."
    )


if __name__ == "__main__":
    main()

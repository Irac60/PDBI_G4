"""
corregir_float32_osdb.py
========================

Corrige una incompatibilidad del toolkit OSDB:

- nnTrainer convierte las muestras del acelerómetro a numpy.float32.
- libosd/dpTools.py intenta serializarlas directamente con json.dumps.
- json.dumps no serializa numpy.float32 sin conversión.

Este parche:
1. Crea una copia de seguridad de dpTools.py.
2. Convierte los valores de rawData y rawData3D a float nativo.
3. Convierte HR a un escalar nativo cuando sea necesario.
4. Es idempotente: si ya fue aplicado, no duplica cambios.

Ejecutar desde la raíz del repositorio:
    python .\corregir_float32_osdb.py
"""

from __future__ import annotations

import shutil
from pathlib import Path


REPO_DIR = Path(r"C:\Users\cesar\osd\OpenSeizureDatabase")
TARGET = REPO_DIR / "libosd" / "dpTools.py"
BACKUP = REPO_DIR / "libosd" / "dpTools_antes_float32.py.bak"


def replace_all(text: str, old: str, new: str) -> tuple[str, int]:
    count = text.count(old)
    return text.replace(old, new), count


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(
            f"No se encontró:\n{TARGET}"
        )

    original = TARGET.read_text(encoding="utf-8")

    # Si ya existe float(...) en las líneas principales, considerar aplicado.
    if (
        "accelLst.append(float(dataObj['rawData'][n]))"
        in original
        and
        "accelLst3d.append(float(dataObj['rawData3D'][n*3]))"
        in original
    ):
        print("El parche ya estaba aplicado.")
        print(f"Archivo: {TARGET}")
        return

    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)
        print(f"Copia de seguridad creada:\n{BACKUP}")

    patched = original
    replacements_total = 0

    replacements = [
        (
            "accelLst.append(dataObj['rawData'][n])",
            "accelLst.append(float(dataObj['rawData'][n]))",
        ),
        (
            "accelLst3d.append(dataObj['rawData3D'][n*3])",
            "accelLst3d.append(float(dataObj['rawData3D'][n*3]))",
        ),
        (
            "accelLst3d.append(dataObj['rawData3D'][n*3 + 1])",
            "accelLst3d.append(float(dataObj['rawData3D'][n*3 + 1]))",
        ),
        (
            "accelLst3d.append(dataObj['rawData3D'][n*3 + 2])",
            "accelLst3d.append(float(dataObj['rawData3D'][n*3 + 2]))",
        ),
    ]

    for old, new in replacements:
        patched, count = replace_all(patched, old, new)
        replacements_total += count

    # Hacer robusta la conversión de HR en el bloque nuevo.
    old_hr = "rawDataObj['HR'] = dataObj['hr']"
    new_hr = (
        "rawDataObj['HR'] = "
        "(dataObj['hr'].item() if hasattr(dataObj['hr'], 'item') "
        "else dataObj['hr'])"
    )
    patched, count = replace_all(patched, old_hr, new_hr)
    replacements_total += count

    # Hacer robusta la conversión de HR en el bloque legado.
    old_hr_legacy = "rawDataObj['HR'] = dataObj['HR']"
    new_hr_legacy = (
        "rawDataObj['HR'] = "
        "(dataObj['HR'].item() if hasattr(dataObj['HR'], 'item') "
        "else dataObj['HR'])"
    )
    patched, count = replace_all(
        patched,
        old_hr_legacy,
        new_hr_legacy,
    )
    replacements_total += count

    if replacements_total == 0:
        raise RuntimeError(
            "No se localizaron las líneas esperadas. "
            "No se modificó dpTools.py."
        )

    TARGET.write_text(patched, encoding="utf-8")

    print("")
    print("=" * 72)
    print("PARCHE APLICADO CORRECTAMENTE")
    print("=" * 72)
    print(f"Archivo modificado:\n{TARGET}")
    print(f"Reemplazos realizados: {replacements_total}")
    print("")
    print("Ahora ejecuta nuevamente:")
    print("python .\\extraer_features_verificar_buffer.py")


if __name__ == "__main__":
    main()

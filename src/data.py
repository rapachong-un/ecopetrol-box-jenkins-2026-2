from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_weekly_returns(path: Path, date_column: str, return_column: str,
                        start_date: str, end_date: str) -> pd.Series:
    if not path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos: {path}. "
            "Confirma que data/raw/ecopetrol_weekly_clean.csv esté incluido."
        )
    frame = pd.read_csv(path, na_values=["NA", "N/A", ""])
    missing = {date_column, return_column} - set(frame.columns)
    if missing:
        raise ValueError(
            f"Faltan las columnas {sorted(missing)}. "
            f"Columnas disponibles: {list(frame.columns)}"
        )
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    if frame[date_column].isna().any():
        raise ValueError("La columna de fechas contiene valores vacíos o inválidos.")
    frame[return_column] = pd.to_numeric(frame[return_column], errors="raise")
    invalid = frame[return_column].isna() & frame[return_column].notna()
    if invalid.any():
        raise ValueError("La columna de retornos contiene valores no numéricos.")
    frame = frame.sort_values(date_column)
    if frame[date_column].duplicated().any():
        raise ValueError("Hay fechas duplicadas en la serie semanal.")
    series = frame.set_index(date_column)[return_column]
    series = series.loc[start_date:end_date].dropna().astype(float)
    if len(series) < 100:
        raise ValueError(f"Se requieren al menos 100 retornos; se encontraron {len(series)}.")
    if not series.index.is_monotonic_increasing:
        raise ValueError("Las fechas no están en orden cronológico.")
    return series.rename("log_return")




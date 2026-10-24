from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (CANDIDATE_ORDERS, DATE_COLUMN, END_DATE, FORECAST_HORIZON,
                        INPUT_FILE, OUTPUT_DIR, RETURN_COLUMN, SELECTED_ORDER, START_DATE)
from src.data import load_weekly_returns
from src.analysis import run_analysis


def main() -> None:
    returns = load_weekly_returns(INPUT_FILE, DATE_COLUMN, RETURN_COLUMN,
                                  START_DATE, END_DATE)
    print(f"Observaciones: {len(returns)}")
    print(f"Periodo observado: {returns.index.min().date()} a {returns.index.max().date()}")
    run_analysis(returns, OUTPUT_DIR, CANDIDATE_ORDERS, SELECTED_ORDER,
                 FORECAST_HORIZON)
    print(f"Resultados guardados en: {OUTPUT_DIR}")
    if SELECTED_ORDER is None:
        print("Identificación y comparación de candidatos listas.")
        print("Interpreta ACF/PACF y documenta manualmente el orden en src/config.py.")
    else:
        print(f"Modelo configurado manualmente: ARIMA{SELECTED_ORDER}")


if __name__ == "__main__":
    main()

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"
FORECAST_HORIZON = 10

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "ecopetrol_weekly_clean.csv"
DATE_COLUMN = "date"
RETURN_COLUMN = "ret"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Se completará manualmente después de interpretar los correlogramas y
# comparar los candidatos documentados. Orden en formato statsmodels (p,d,q).
SELECTED_ORDER = (2, 0, 0)
CANDIDATE_ORDERS = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1),
                    (2, 0, 0), (0, 0, 2), (2, 0, 1), (1, 0, 2)]


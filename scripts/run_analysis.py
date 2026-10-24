"""Análisis Box–Jenkins de los log retornos semanales de Ecopetrol (BVC)."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


# Todas las rutas se calculan desde la ubicación de este archivo.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "ecopetrol_weekly_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

START_DATE = "2010-01-01"
END_DATE = "2025-12-31"
DATE_COLUMN = "date"
RETURN_COLUMN = "ret"
FORECAST_HORIZON = 10

# Lista fija para comparar modelos motivados por la identificación.
CANDIDATE_ORDERS = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 0, 1),
    (1, 0, 1),
    (2, 0, 0),
    (0, 0, 2),
    (2, 0, 1),
    (1, 0, 2),
]

# Orden ARIMA(2, 0, 0), fijado manualmente después de interpretar ACF/PACF
# y comparar los candidatos. El programa no elige el orden automáticamente.
SELECTED_ORDER = (2, 0, 0)


def load_weekly_returns() -> pd.Series:
    """Lee y valida los log retornos semanales del archivo incluido."""
    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos: {DATA_FILE}. "
            "Debe estar incluido en data/raw/ecopetrol_weekly_clean.csv."
        )

    frame = pd.read_csv(DATA_FILE, na_values=["NA", "N/A", ""])
    required_columns = {DATE_COLUMN, RETURN_COLUMN}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Faltan las columnas {sorted(missing_columns)}. "
            f"Columnas disponibles: {list(frame.columns)}."
        )

    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    if frame[DATE_COLUMN].isna().any():
        raise ValueError("La columna de fechas contiene valores vacíos o inválidos.")

    try:
        frame[RETURN_COLUMN] = pd.to_numeric(frame[RETURN_COLUMN], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("La columna de retornos contiene valores no numéricos.") from exc

    frame = frame.sort_values(DATE_COLUMN)
    if frame[DATE_COLUMN].duplicated().any():
        raise ValueError("Hay fechas duplicadas en la serie semanal.")

    returns = (
        frame.set_index(DATE_COLUMN)[RETURN_COLUMN]
        .loc[START_DATE:END_DATE]
        .dropna()
        .astype(float)
    )
    if len(returns) < 100:
        raise ValueError(
            f"Se requieren al menos 100 observaciones; se encontraron {len(returns)}."
        )
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("La columna de retornos contiene valores infinitos.")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("Las fechas no están en orden cronológico.")

    return returns.rename("log_return")


def save_identification_plot(returns: pd.Series) -> None:
    """Guarda la serie, su distribución y los correlogramas ACF/PACF."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    returns.plot(ax=axes[0, 0], color="#176B87", linewidth=0.8)
    axes[0, 0].set(
        title="Log retornos semanales de Ecopetrol",
        xlabel="",
        ylabel="Log retorno",
        xlim=(returns.index.min(), returns.index.max()),
    )
    returns.hist(ax=axes[0, 1], bins=35, color="#64CCC5", edgecolor="white")
    axes[0, 1].set(
        title="Distribución de retornos",
        xlabel="Log retorno",
        ylabel="Frecuencia",
    )
    plot_acf(returns, lags=30, ax=axes[1, 0], zero=False)
    axes[1, 0].set_title("ACF de los retornos")
    plot_pacf(returns, lags=30, ax=axes[1, 1], method="ywm", zero=False)
    axes[1, 1].set_title("PACF de los retornos")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "identificacion.png", dpi=180)
    plt.close(fig)


def compare_candidate_models(returns: pd.Series) -> None:
    """Estima una lista fija de candidatos y guarda AIC/BIC para revisión."""
    rows = []
    for order in CANDIDATE_ORDERS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ARIMA(returns.to_numpy(), order=order, trend="c").fit()
            mle_results = getattr(fitted, "mle_retvals", {})
            rows.append(
                {
                    "p": order[0],
                    "d": order[1],
                    "q": order[2],
                    "aic": fitted.aic,
                    "bic": fitted.bic,
                    "convergio": bool(mle_results.get("converged", True)),
                    "loglik": fitted.llf,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "p": order[0],
                    "d": order[1],
                    "q": order[2],
                    "aic": np.nan,
                    "bic": np.nan,
                    "convergio": False,
                    "loglik": np.nan,
                    "error": str(exc),
                }
            )

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "comparacion_candidatos.csv", index=False
    )


def save_model_diagnostics(model, returns: pd.Series) -> None:
    """Guarda pruebas y gráficas para revisar los residuos del ARIMA."""
    residuals = pd.Series(
        np.asarray(model.resid), index=returns.index, name="residuo"
    ).dropna()

    ljung_box = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
    arch_test = het_arch(residuals, nlags=10, result_object=False)
    jb_test = jarque_bera(residuals)

    diagnostics = {
        "orden": str(SELECTED_ORDER),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "ljung_box_p_10": float(ljung_box.loc[10, "lb_pvalue"]),
        "ljung_box_p_20": float(ljung_box.loc[20, "lb_pvalue"]),
        "arch_lm_p": float(arch_test[1]),
        "jarque_bera_p": float(jb_test[1]),
    }
    (OUTPUT_DIR / "diagnosticos_modelo.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    residuals.to_frame().to_csv(
        OUTPUT_DIR / "residuos.csv", index_label="date"
    )

    fig = model.plot_diagnostics(figsize=(11, 8))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "diagnostico_residuos.png", dpi=180)
    plt.close(fig)


def save_forecast(model, returns: pd.Series) -> None:
    """Guarda el pronóstico a diez semanas y su gráfica con intervalo al 95 %."""
    forecast = model.get_forecast(steps=FORECAST_HORIZON)
    first_future_day = returns.index[-1] + pd.Timedelta(days=1)
    future_dates = pd.date_range(
        first_future_day, periods=FORECAST_HORIZON, freq="W-FRI"
    )
    confidence_interval = np.asarray(forecast.conf_int())

    table = pd.DataFrame(
        {
            "fecha": future_dates,
            "pronostico_log_retorno": np.asarray(forecast.predicted_mean),
            "limite_inferior_95": confidence_interval[:, 0],
            "limite_superior_95": confidence_interval[:, 1],
        }
    )
    table.to_csv(OUTPUT_DIR / "pronostico_10_semanas.csv", index=False)

    observed_dates = mdates.date2num(returns.index[-80:].to_pydatetime())
    future_x = mdates.date2num(future_dates.to_pydatetime())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(
        observed_dates,
        returns.iloc[-80:].to_numpy(),
        color="#176B87",
        label="Observado",
    )
    ax.plot(
        future_x,
        table["pronostico_log_retorno"].to_numpy(),
        color="#D9534F",
        marker="o",
        label="Pronóstico",
    )
    ax.fill_between(
        future_x,
        table["limite_inferior_95"].to_numpy(),
        table["limite_superior_95"].to_numpy(),
        color="#D9534F",
        alpha=0.18,
        label="Intervalo 95 %",
    )
    ax.xaxis_date()
    ax.set(
        title=f"Pronóstico semanal, ARIMA{SELECTED_ORDER}",
        xlabel="Fecha",
        ylabel="Log retorno",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pronostico.png", dpi=180)
    plt.close(fig)


def run_analysis(returns: pd.Series) -> None:
    """Ejecuta Box–Jenkins y genera todos los resultados en outputs/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    returns.to_frame().to_csv(
        OUTPUT_DIR / "serie_analizada.csv", index_label="date"
    )
    returns.describe().rename("valor").to_csv(
        OUTPUT_DIR / "estadisticas_descriptivas.csv"
    )
    save_identification_plot(returns)

    adf = adfuller(
        returns,
        regression="c",
        maxlag=12,
        autolag=None,
        result_object=False,
    )
    adf_result = pd.DataFrame(
        [
            {
                "estadistico_adf": adf[0],
                "valor_p": adf[1],
                "rezagos_fijos": 12,
                "observaciones": adf[3],
                **{f"critico_{key}": value for key, value in adf[4].items()},
            }
        ]
    )
    adf_result.to_csv(OUTPUT_DIR / "prueba_adf.csv", index=False)

    compare_candidate_models(returns)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(
            returns.to_numpy(), order=SELECTED_ORDER, trend="c"
        ).fit()

    save_model_diagnostics(model, returns)
    save_forecast(model, returns)


def main() -> None:
    returns = load_weekly_returns()
    print(f"Observaciones: {len(returns)}")
    print(
        "Periodo observado: "
        f"{returns.index.min().date()} a {returns.index.max().date()}"
    )
    run_analysis(returns)
    print(f"Resultados guardados en: {OUTPUT_DIR}")
    print(f"Orden ARIMA fijado manualmente: {SELECTED_ORDER}")


if __name__ == "__main__":
    main()

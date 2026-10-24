from __future__ import annotations

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera


def run_analysis(returns: pd.Series, outdir: Path, candidate_orders: list[tuple],
                 selected_order: tuple | None, horizon: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    returns.to_frame().to_csv(outdir / "serie_analizada.csv", index_label="date")
    returns.describe().rename("valor").to_csv(outdir / "estadisticas_descriptivas.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    returns.plot(ax=axes[0, 0], color="#176B87", lw=0.8)
    axes[0, 0].set(title="Log retornos semanales de Ecopetrol", xlabel="", ylabel="Log retorno", xlim=(returns.index.min(), returns.index.max()))
    returns.hist(ax=axes[0, 1], bins=35, color="#64CCC5", edgecolor="white")
    axes[0, 1].set(title="Distribución de retornos", xlabel="Log retorno", ylabel="Frecuencia")
    plot_acf(returns, lags=30, ax=axes[1, 0], zero=False)
    axes[1, 0].set_title("ACF de los retornos")
    plot_pacf(returns, lags=30, ax=axes[1, 1], method="ywm", zero=False)
    axes[1, 1].set_title("PACF de los retornos")
    fig.tight_layout()
    fig.savefig(outdir / "identificacion.png", dpi=180)
    plt.close(fig)

    adf = adfuller(returns, regression="c", maxlag=12, autolag=None, result_object=False)
    pd.DataFrame([{
        "estadistico_adf": adf[0], "valor_p": adf[1], "rezagos_fijos": 12,
        "observaciones": adf[3], **{f"critico_{k}": v for k, v in adf[4].items()}
    }]).to_csv(outdir / "prueba_adf.csv", index=False)

    rows = []
    for order in candidate_orders:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ARIMA(returns.to_numpy(), order=order, trend="c").fit()
            rows.append({
                "p": order[0], "d": order[1], "q": order[2],
                "aic": fitted.aic, "bic": fitted.bic,
                "convergio": bool(fitted.mle_retvals.get("converged", True)),
                "loglik": fitted.llf
            })
        except Exception as exc:
            rows.append({"p": order[0], "d": order[1], "q": order[2],
                         "aic": np.nan, "bic": np.nan, "convergio": False,
                         "loglik": np.nan, "error": str(exc)})
    pd.DataFrame(rows).to_csv(outdir / "comparacion_candidatos.csv", index=False)

    if selected_order is None:
        return
    model = ARIMA(returns.to_numpy(), order=selected_order, trend="c").fit()
    resid = pd.Series(model.resid, index=returns.index).dropna()
    lb = acorr_ljungbox(resid, lags=[10, 20], return_df=True)
    arch = het_arch(resid, nlags=10, result_object=False)
    jb = jarque_bera(resid)
    diagnostic = {
        "orden": str(selected_order), "aic": model.aic, "bic": model.bic,
        "ljung_box_p_10": float(lb.loc[10, "lb_pvalue"]),
        "ljung_box_p_20": float(lb.loc[20, "lb_pvalue"]),
        "arch_lm_p": float(arch[1]), "jarque_bera_p": float(jb[1])
    }
    (outdir / "diagnosticos_modelo.json").write_text(
        json.dumps(diagnostic, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame({"residuo": resid}).to_csv(outdir / "residuos.csv", index_label="date")

    fig = model.plot_diagnostics(figsize=(11, 8))
    fig.tight_layout()
    fig.savefig(outdir / "diagnostico_residuos.png", dpi=180)
    plt.close(fig)

    forecast = model.get_forecast(steps=horizon)
    future_dates = pd.date_range(returns.index[-1] + pd.Timedelta(days=1),
                                periods=horizon, freq="W-FRI")
    table = pd.DataFrame({
        "fecha": future_dates,
        "pronostico_log_retorno": np.asarray(forecast.predicted_mean),
        "limite_inferior_95": np.asarray(forecast.conf_int()[:, 0]),
        "limite_superior_95": np.asarray(forecast.conf_int()[:, 1])
    })
    table.to_csv(outdir / "pronostico_10_semanas.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    import matplotlib.dates as mdates
    observed_dates = mdates.date2num(returns.index[-80:].to_pydatetime())
    future_x = mdates.date2num(future_dates.to_pydatetime())
    ax.plot(observed_dates, returns.iloc[-80:].to_numpy(),
            color="#176B87", label="Observado")
    ax.plot(future_x, table["pronostico_log_retorno"].to_numpy(),
            color="#D9534F", marker="o", label="Pronóstico")
    ax.fill_between(future_x, table["limite_inferior_95"].to_numpy(),
                    table["limite_superior_95"].to_numpy(), color="#D9534F",
                    alpha=0.18, label="Intervalo 95%")
    ax.xaxis_date()
    ax.set(title=f"Pronóstico semanal, ARIMA{selected_order}",
           xlabel="Fecha", ylabel="Log retorno")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "pronostico.png", dpi=180)
    plt.close(fig)






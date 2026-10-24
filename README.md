# Retornos semanales de Ecopetrol (BVC): Box–Jenkins

Proyecto reproducible para analizar los log retornos semanales de la acción ordinaria de Ecopetrol en la Bolsa de Valores de Colombia, entre 2010 y 2025.

## Datos y muestra

El análisis usa el archivo incluido en data/raw/ecopetrol_weekly_clean.csv. La variable ret contiene los log retornos semanales existentes en el archivo; el código valida fechas, valores y duplicados, pero no vuelve a calcular retornos desde precios.

El filtro de estudio cubre del 1 de enero de 2010 al 31 de diciembre de 2025. El CSV contiene 834 observaciones semanales, desde el 15 de enero de 2010 hasta el 30 de diciembre de 2025. La prueba ADF usa 821 observaciones efectivas al aplicar 12 rezagos. El experimento exploratorio de 2022–2025 no se incluye.

## Modelo y resultados

Se aplica la metodología Box–Jenkins: identificación mediante la serie y los correlogramas ACF/PACF; contraste de estacionariedad ADF con 12 rezagos fijos; estimación y comparación de una lista predeterminada de modelos; diagnóstico de residuos; y pronóstico a diez semanas.

El código estima los ocho candidatos fijados en el propio archivo. No realiza selección automática. El orden usado para el pronóstico es ARIMA(2, 0, 0), fijado manualmente tras interpretar la identificación y comparar los candidatos.

En los resultados actuales, el ADF es −7.870 (p < 0.001), lo que permite rechazar la hipótesis de raíz unitaria. Para ARIMA(2, 0, 0), AIC = −2737.89 y BIC = −2718.98. Ljung–Box da p = 0.510 a 10 rezagos y p = 0.715 a 20, sin evidencia de autocorrelación residual en esos rezagos. La prueba ARCH da p < 0.001 y Jarque–Bera p < 0.001: permanecen heterocedasticidad y no normalidad en los residuos. Por tanto, el AR(2) describe la media condicional, pero no modela por sí solo la volatilidad; los intervalos del pronóstico deben interpretarse con cautela.

## Ejecutar

Requiere Python 3.12. Desde la carpeta principal del repositorio:

Windows PowerShell:

    python -m venv .venv
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    .venv\Scripts\python.exe scripts/run_analysis.py

macOS/Linux:

    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python scripts/run_analysis.py

## Archivos y resultados

Todo el código Python necesario está en scripts/run_analysis.py; ese es el único archivo que se debe ejecutar. El script resuelve las rutas desde su propia ubicación, lee el CSV incluido y crea outputs/ al ejecutarse.

La carpeta outputs/ contiene tablas y figuras regenerables: descriptivos, ADF, comparación de candidatos, diagnósticos residuales, y pronóstico a diez semanas. Se excluye del repositorio para mantenerlo compacto; el script la vuelve a crear cuando se ejecuta.

El orden ARIMA se fija manualmente en la constante SELECTED_ORDER al comienzo del archivo. La lista CANDIDATE_ORDERS también se define allí y solo se compara; el programa no escoge automáticamente un modelo.

# Retornos semanales de Ecopetrol (BVC): Box–Jenkins

Proyecto reproducible para analizar los log retornos semanales de la acción ordinaria de Ecopetrol en la Bolsa de Valores de Colombia, entre 2010 y 2025.

## Datos y muestra

El análisis usa `data/raw/ecopetrol_weekly_clean.csv`. La variable `ret` contiene los log retornos semanales; el código valida fechas, valores y duplicados, pero no recalcula retornos desde precios.

El filtro cubre del 1 de enero de 2010 al 31 de diciembre de 2025. El CSV contiene 834 observaciones, desde el 15 de enero de 2010 hasta el 30 de diciembre de 2025. La prueba ADF con intercepto selecciona 2 rezagos mediante AIC y usa 831 observaciones efectivas. El recorte exploratorio 2022–2025 no se incluye. La validación retrospectiva separada entrena con 793 observaciones hasta el 20 de marzo de 2025 y evalúa las 41 semanas restantes del año.

## Modelo y resultados

Se aplica Box–Jenkins: identificación con la serie y ACF/PACF; prueba ADF; estimación y comparación de candidatos predeterminados; diagnóstico de residuos; validación retrospectiva frente al retorno cero; y pronóstico final a diez semanas.

El modelo para la media es ARIMA(2, 0, 0), equivalente a AR(2), fijado manualmente. Los candidatos se comparan en una lista explícita; el programa no selecciona automáticamente el orden ARMA. En el contraste ADF, AIC selecciona 2 rezagos. Para Ljung–Box de los residuos del AR(2), se ajustan 2 grados de libertad por los parámetros autorregresivos estimados.

Resultados reproducibles: ADF = −18.378 (p < 0.001); AR(2) AIC = −2737.89 y BIC = −2718.98. Ljung–Box ajustado da p = 0.134, 0.323 y 0.590 a 5, 10 y 20 rezagos. ARCH-LM da p ≈ 1.06 × 10⁻¹⁵ y Jarque–Bera p < 0.001: persisten heterocedasticidad y no normalidad. En la validación de 41 semanas, el RMSE del AR(2) es 0.03942 frente a 0.03950 del pronóstico cero, una mejora de aproximadamente 0.2 %. El AR(2) describe la media condicional, pero no modela la volatilidad; los intervalos deben interpretarse con cautela.

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

Todo el código Python está en `scripts/run_analysis.py`, el único archivo que se debe ejecutar. Las rutas se resuelven desde la ubicación del script. El análisis crea o actualiza `outputs/`.

Los resultados incluidos y regenerables abarcan descriptivos, prueba ADF, tabla de parámetros, comparación de candidatos, diagnósticos residuales, validación fuera de muestra de 41 semanas, serie analizada y pronóstico final a diez semanas.

El póster final está en `poster/ecopetrol_box_jenkins_poster.pdf`, con su vista previa PNG en la misma carpeta.

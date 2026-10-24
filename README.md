# Retornos semanales de Ecopetrol (BVC): Box–Jenkins

Proyecto reproducible para analizar los log retornos semanales de la acción ordinaria de Ecopetrol en la Bolsa de Valores de Colombia, entre 2010 y 2025.

## Ejecutar en una copia descargada

Requiere Python 3.12.

En Windows PowerShell:

    python -m venv .venv
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    .venv\Scripts\python.exe scripts/run_analysis.py

En macOS/Linux:

    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python scripts/run_analysis.py

El script localiza datos y resultados con rutas relativas a esta carpeta. El CSV está incluido en data/raw/. Ejecuta identificación, compara la lista fija de candidatos, estima el orden configurado manualmente y genera el pronóstico a diez semanas.
## Contenido

- data/raw/: serie semanal original usada en el estudio.
- src/: lectura, validación, estimación y diagnósticos.
- scripts/run_analysis.py: punto de entrada reproducible.
- outputs/: tablas y gráficas regeneradas por el script.
- docs/: decisiones metodológicas del análisis.
- poster/: destino del póster final.

El orden ARIMA se fija manualmente en src/config.py después de interpretar ACF/PACF y justificar la estimación. El código no selecciona automáticamente un modelo. El análisis cubre la muestra completa 2010–2025 y no contiene el experimento 2022–2025.

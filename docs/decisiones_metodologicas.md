# Decisiones metodológicas

## Serie de estudio

- Activo: acción ordinaria de Ecopetrol negociada en la Bolsa de Valores de Colombia.
- Frecuencia: semanal, usando el último día de cotización de cada semana.
- Muestra completa: 2010-01-01 a 2025-12-31.
- Variable: log retorno semanal existente en el CSV (`ret`), sujeto a confirmación de fórmula y tratamiento de precios ajustados.
- No se incluirá el experimento exploratorio con una submuestra 2022-2025.

## Box–Jenkins

Se analizará la muestra completa, en orden: identificación, estimación, validación y pronóstico a diez periodos. Los órdenes se justificarán manualmente a partir de los correlogramas y la comparación de candidatos previamente motivados. No se incorporarán procedimientos de selección automática ni sus resultados.

Se evaluarán autocorrelación residual, heterocedasticidad, distribución de residuos y otros supuestos pertinentes. Toda limitación encontrada se reportará, especialmente porque un ARMA/ARIMA de la media no modela por sí solo la volatilidad condicional.

## Reproducibilidad

El script utilizará rutas relativas basadas en la ubicación del proyecto. El archivo fuente original se conservará en `data/raw/`; las tablas y figuras finales se volverán a generar desde ese archivo y el código depurado.

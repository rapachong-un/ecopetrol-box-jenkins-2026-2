from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.pdfbase.pdfmetrics import stringWidth
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "poster" / "ecopetrol_box_jenkins_poster.pdf"
PREVIEW = ROOT / "poster" / "ecopetrol_box_jenkins_poster.png"
RESULTS = ROOT / "outputs"
DATA = ROOT / "data" / "raw" / "ecopetrol_weekly_clean.csv"

PAGE_W, PAGE_H = 32 * 72, 22 * 72
NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#176B87")
TEAL = colors.HexColor("#19A7A0")
GOLD = colors.HexColor("#E7A33E")
INK = colors.HexColor("#172B4D")
MUTED = colors.HexColor("#52667A")
PALE = colors.HexColor("#F3F7FA")
LINE = colors.HexColor("#D7E1EA")
WHITE = colors.white
RED = colors.HexColor("#BC4B51")
GREEN = colors.HexColor("#1F7A62")

styles = {
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=17,
                           leading=23, textColor=INK, spaceAfter=8),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=13.5,
                            leading=18, textColor=MUTED),
    "body_white": ParagraphStyle("body_white", fontName="Helvetica", fontSize=17,
                                 leading=23, textColor=WHITE),
    "caption": ParagraphStyle("caption", fontName="Helvetica", fontSize=13,
                              leading=16, textColor=MUTED),
    "center": ParagraphStyle("center", fontName="Helvetica", fontSize=15,
                             leading=20, textColor=INK, alignment=TA_CENTER),
}

def para(c, text, x, top, width, style="body"):
    p = Paragraph(text, styles[style])
    _, h = p.wrap(width, PAGE_H)
    p.drawOn(c, x, PAGE_H - top - h)
    return h

def panel(c, x, top, w, h, title, accent=BLUE):
    y = PAGE_H - top - h
    c.setFillColor(WHITE)
    c.setStrokeColor(LINE)
    c.setLineWidth(1.2)
    c.roundRect(x, y, w, h, 16, fill=1, stroke=1)
    c.setFillColor(accent)
    c.roundRect(x, PAGE_H-top-58, w, 58, 16, fill=1, stroke=0)
    c.rect(x, PAGE_H-top-58, w, 18, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(x+24, PAGE_H-top-38, title)

def image_fit(c, path, x, top, w, h):
    im = ImageReader(str(path))
    iw, ih = im.getSize()
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    c.drawImage(im, x+(w-dw)/2, PAGE_H-top-dh, width=dw, height=dh,
                preserveAspectRatio=True, mask="auto")

def metric(c, x, top, w, label, value, note=""):
    c.setFillColor(BLUE)
    c.roundRect(x, PAGE_H-top-88, w, 88, 13, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#C5D8E6"))
    c.setFont("Helvetica-Bold", 13)
    c.drawString(x+17, PAGE_H-top-25, label.upper())
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 28)
    c.drawString(x+17, PAGE_H-top-58, value)
    if note:
        c.setFillColor(colors.HexColor("#D7EAF2"))
        c.setFont("Helvetica", 11)
        c.drawRightString(x+w-15, PAGE_H-top-58, note)

def draw_table(c, x, top, widths, rows, row_h=34, header=True):
    y = PAGE_H-top
    total = sum(widths)
    for ri, row in enumerate(rows):
        yy = y - (ri+1)*row_h
        if ri == 0 and header:
            c.setFillColor(NAVY)
            c.rect(x, yy, total, row_h, fill=1, stroke=0)
            c.setFillColor(WHITE)
            font = "Helvetica-Bold"
            fs = 13
        else:
            c.setFillColor(PALE if ri % 2 else WHITE)
            c.rect(x, yy, total, row_h, fill=1, stroke=0)
            c.setFillColor(INK)
            font = "Helvetica-Bold" if ri == len(rows)-1 else "Helvetica"
            fs = 13.5
        xx = x
        for ci, cell in enumerate(row):
            c.setFont(font, fs)
            if ci == 0:
                c.drawString(xx+9, yy+11, str(cell))
            else:
                c.drawRightString(xx+widths[ci]-9, yy+11, str(cell))
            xx += widths[ci]
    c.setStrokeColor(LINE)
    c.rect(x, y-len(rows)*row_h, total, len(rows)*row_h, fill=0, stroke=1)

def main():
    data = pd.read_csv(DATA, na_values=["NA"])
    valid = data.dropna(subset=["ret"])
    n = len(valid)
    start, end = valid.date.iloc[0], valid.date.iloc[-1]
    stats = valid.ret.astype(float)
    diag = json.loads((RESULTS / "diagnosticos_modelo.json").read_text(encoding="utf-8"))
    fc = pd.read_csv(RESULTS / "pronostico_10_semanas.csv")
    forecast_mean = fc.pronostico_log_retorno.mean() * 100
    adf = pd.read_csv(RESULTS / "prueba_adf.csv").iloc[0]
    candidates = pd.read_csv(RESULTS / "comparacion_candidatos.csv")
    def bic(p, q):
        row = candidates[(candidates.p == p) & (candidates.q == q)].iloc[0]
        return float(row.bic)

    c = canvas.Canvas(str(OUT), pagesize=(PAGE_W, PAGE_H))
    c.setTitle("¿Memoria predecible o ruido blanco? Ecopetrol BVC")
    c.setAuthor("Trabajo 1 - Econometría II")
    c.setFillColor(PALE)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    # Header
    c.setFillColor(NAVY)
    c.rect(0, PAGE_H-166, PAGE_W, 166, fill=1, stroke=0)
    c.setFillColor(TEAL)
    c.rect(0, PAGE_H-166, 15, 166, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 42)
    c.drawString(58, PAGE_H-68, "¿MEMORIA PREDECIBLE O RUIDO BLANCO?")
    c.setFont("Helvetica", 23)
    c.setFillColor(colors.HexColor("#D7EAF2"))
    c.drawString(60, PAGE_H-108, "Log retornos semanales de Ecopetrol · Bolsa de Valores de Colombia")
    c.setFont("Helvetica", 15)
    c.setFillColor(colors.HexColor("#B9D2DF"))
    c.drawString(60, PAGE_H-140, "Un modelo AR(2) para la media y pruebas de diagnóstico, 2010–2025")

    # Metrics strip
    x0, gap = 50, 18
    mw = (PAGE_W-100-gap*3)/4
    topm = 190
    metric(c, x0, topm, mw, "Observaciones", f"{n}", f"{start} — {end}")
    metric(c, x0+(mw+gap), topm, mw, "Retorno medio", f"{stats.mean()*100:.3f}%", "por semana")
    metric(c, x0+2*(mw+gap), topm, mw, "Desviación estándar", f"{stats.std()*100:.2f}%", "volatilidad semanal")
    metric(c, x0+3*(mw+gap), topm, mw, "Pronóstico medio h=1…10", f"{forecast_mean:.3f}%", "log retorno/semana")

    # Three columns
    margin, gap = 50, 24
    colw = (PAGE_W-2*margin-2*gap)/3
    xs = [margin, margin+colw+gap, margin+2*(colw+gap)]
    top, panel_h = 300, 1115

    # Column 1: motivation/data and plots
    panel(c, xs[0], top, colw, panel_h, "PREGUNTA Y DATOS", BLUE)
    y = top+82
    y += para(c, "<b>Motivación económica.</b> En la forma débil de la hipótesis de mercados eficientes, los retornos pasados no deberían ofrecer una regla lineal estable para anticipar retornos futuros. Evaluamos si hay memoria lineal en la media; esto no constituye, por sí solo, una prueba completa de eficiencia del mercado.", xs[0]+24, y, colw-48, "body")
    y += 18
    y += para(c, f"<b>Serie.</b> Acción ordinaria de Ecopetrol negociada en la BVC; última cotización disponible de cada semana. Muestra: {n} log retornos, {start} a {end}.", xs[0]+24, y, colw-48, "body")
    y += 14
    y += para(c, "<b>Transformación.</b> r_t = ln(P_t/P_t-1). La columna de retornos coincide numéricamente con la primera diferencia del log cierre (máxima diferencia: 2.6e-15).", xs[0]+24, y, colw-48, "body")
    y += 18
    y += para(c, "<b>Motivo del ARMA.</b> Los log retornos evitan modelar directamente una serie de precios no estacionaria y permiten preguntar si la historia reciente contiene información para la media condicional.", xs[0]+24, y, colw-48, "small")
    y += 20
    c.setFillColor(MUTED)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(xs[0]+24, PAGE_H-y-18, "COMPORTAMIENTO Y CORRELOGRAMAS")
    y += 30
    image_fit(c, RESULTS / "identificacion.png", xs[0]+16, y, colw-32, 410)
    y += 420
    para(c, "La ACF y la PACF muestran señales en los primeros rezagos; su patrón motiva evaluar candidatos ARMA de bajo orden.", xs[0]+24, y, colw-48, "caption")

    # Column 2: methodology and model comparison
    panel(c, xs[1], top, colw, panel_h, "BOX–JENKINS", TEAL)
    y = top+83
    steps = [
        ("1 · IDENTIFICACIÓN", "ADF con 12 rezagos fijos: estadístico %.2f, p < 0.001; se rechaza raíz unitaria para la media." % float(adf.estadistico_adf)),
        ("2 · ESTIMACIÓN", "Se estimaron candidatos ARMA de bajo orden motivados por ACF/PACF. Orden elegido manualmente: ARMA(2,0), equivalente a ARIMA(2,0,0)."),
        ("3 · SELECCIÓN RAZONADA", "Entre los candidatos predefinidos, ARMA(2,0) presenta BIC %.2f; ARMA(0,0) presenta BIC %.2f (cambio BIC = %.2f). La ventaja BIC frente al ruido blanco es pequeña." % (bic(2,0), bic(0,0), bic(2,0)-bic(0,0))),
        ("4 · VALIDACIÓN", "Ljung–Box sobre residuos a 10 y 20 rezagos; ARCH–LM y Jarque–Bera para examinar varianza y distribución."),
        ("5 · PRONÓSTICO", "Diez semanas adelante, con intervalos de predicción al 95%%. El pronóstico se refiere a la media; no modela la volatilidad condicional.")
    ]
    for title, body in steps:
        c.setFillColor(BLUE if title.startswith(("1","2","3")) else TEAL)
        c.circle(xs[1]+37, PAGE_H-y-11, 11, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(xs[1]+37, PAGE_H-y-15, title.split(" ")[0].replace("·",""))
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 17)
        c.drawString(xs[1]+61, PAGE_H-y-7, title)
        y += 17
        h = para(c, body, xs[1]+61, y, colw-85, "small")
        y += h+20
    y += 4
    c.setStrokeColor(LINE)
    c.line(xs[1]+24, PAGE_H-y, xs[1]+colw-24, PAGE_H-y)
    y += 24
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 17)
    c.drawString(xs[1]+24, PAGE_H-y-16, "CANDIDATOS (BIC)")
    y += 24
    rows = [["Modelo", "BIC", "Cambio BIC"],
            ["ARMA(2,0)", f"{bic(2,0):.2f}", "0.00"],
            ["ARMA(0,0)", f"{bic(0,0):.2f}", f"{bic(0,0)-bic(2,0):.2f}"],
            ["ARMA(1,0)", f"{bic(1,0):.2f}", f"{bic(1,0)-bic(2,0):.2f}"],
            ["ARMA(0,1)", f"{bic(0,1):.2f}", f"{bic(0,1)-bic(2,0):.2f}"],
            ["ARMA(1,1)", f"{bic(1,1):.2f}", f"{bic(1,1)-bic(2,0):.2f}"]]
    draw_table(c, xs[1]+24, y+4, [195, 195, 190], rows, row_h=32)
    y += len(rows)*32+28
    para(c, "La búsqueda es manual y acotada; no se usa selección automática. BIC menor es preferible, pero la diferencia de 0.91 frente a ARMA(0,0) no representa evidencia fuerte por sí sola.", xs[1]+24, y, colw-48, "caption")

    # Column 3: validation, forecast, conclusion
    panel(c, xs[2], top, colw, panel_h, "RESULTADOS Y ALCANCE", GOLD)
    y = top+84
    validation = [
        ("Media AR(2)", "AR(1) = 0.079 (p < 0.001); AR(2) = -0.110 (p < 0.001)."),
        ("Residuos: media", "Ljung–Box p = %.3f (10 rezagos) y %.3f (20): no se detecta autocorrelación residual." % (diag["ljung_box_p_10"], diag["ljung_box_p_20"])),
        ("Residuos: varianza", "ARCH–LM p < 0.001: persiste heterocedasticidad condicional."),
        ("Residuos: distribución", "Jarque–Bera p < 0.001: se rechaza normalidad.")
    ]
    for label, text in validation:
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(xs[2]+24, PAGE_H-y-17, label.upper())
        y += 25
        h = para(c, text, xs[2]+24, y, colw-48, "small")
        y += h+14
    c.setFillColor(MUTED)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(xs[2]+24, PAGE_H-y-12, "PRONÓSTICO DE 10 SEMANAS")
    y += 24
    image_fit(c, RESULTS / "pronostico.png", xs[2]+20, y, colw-40, 235)
    y += 242
    forecast_rows = [["Semana", "Pronóstico", "LI 95%", "LS 95%"]]
    for _, row in fc.iterrows():
        forecast_rows.append([
            pd.Timestamp(row.fecha).strftime("%d %b %y"),
            f"{row.pronostico_log_retorno*100:.2f}%",
            f"{row.limite_inferior_95*100:.2f}%",
            f"{row.limite_superior_95*100:.2f}%"
        ])
    draw_table(c, xs[2]+24, y+2, [150, 145, 145, 145], forecast_rows, row_h=22)
    y += len(forecast_rows)*22+20
    para(c, "La media pronosticada converge hacia cero y los intervalos son amplios; no se modela la volatilidad condicional.", xs[2]+24, y, colw-48, "caption")
    y += 58
    c.setFillColor(PALE)
    c.roundRect(xs[2]+20, PAGE_H-y-150, colw-40, 150, 12, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 17)
    c.drawString(xs[2]+38, PAGE_H-y-28, "CONCLUSIÓN")
    para(c, "El AR(2) capta dependencia lineal débil en los primeros rezagos, pero la evidencia frente al ruido blanco es modesta. Los residuos no muestran autocorrelación, aunque sí agrupamiento de volatilidad y colas no normales. Por tanto, el modelo describe la media de forma limitada y no basta para caracterizar el riesgo.", xs[2]+38, y+40, colw-76, "small")

    # Footer / sources
    footer_top = 1460
    c.setFillColor(NAVY)
    c.rect(0, 0, PAGE_W, PAGE_H-footer_top, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 70, "FUENTES Y NOTAS")
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.HexColor("#D6E5ED"))
    c.drawString(50, 48, "Datos: base semanal ecopetrol_weekly_clean.csv suministrada por el equipo; columna ret. Frecuencia: último cierre BVC de cada semana.")
    c.drawString(50, 29, "Motivación: Fama, E. F. (1970), Efficient Capital Markets, Journal of Finance 25(2), 383–417. doi:10.1111/j.1540-6261.1970.tb00518.x")
    c.drawRightString(PAGE_W-50, 48, "Econometría II · Trabajo 1 · 2026-2")
    c.drawRightString(PAGE_W-50, 29, "Formato 32 × 22 in · diagonal 38.8 in (máximo permitido: 40 in)")
    c.save()

if __name__ == "__main__":
    main()



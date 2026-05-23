import os
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import tensorflow as tf

# ─── 1. CARGA DE DATOS Y MODELOS ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(BASE_DIR, "data", "saber11_features.csv"))

model_p1 = tf.keras.models.load_model(os.path.join(BASE_DIR, "models", "model_p1.keras"))
model_p2 = tf.keras.models.load_model(os.path.join(BASE_DIR, "models", "model_p2.keras"))
model_p3 = tf.keras.models.load_model(os.path.join(BASE_DIR, "models", "model_p3.keras"))

# ─── 2. FEATURES Y SCALERS ───────────────────────────────────────────────────

FEAT_P1 = [
    'estrato_num', 'edu_madre_num', 'edu_padre_num', 'indice_activos',
    'fami_tienecomputador', 'fami_tieneinternet',
    'fami_tieneautomovil', 'fami_tienelavadora',
    'es_privado', 'es_urbano', 'genero_num'
]
FEAT_P2 = FEAT_P1

FEAT_P3 = [
    'punt_matematicas', 'punt_lectura_critica',
    'punt_c_naturales', 'punt_sociales_ciudadanas',
    'estrato_num', 'edu_madre_num', 'edu_padre_num', 'indice_activos',
    'fami_tienecomputador', 'fami_tieneinternet',
    'fami_tieneautomovil', 'fami_tienelavadora',
    'es_privado', 'es_urbano', 'cole_bilingue', 'genero_num'
]

# Orden alfabético = orden que usó LabelEncoder en el entrenamiento
CLASES_P3 = ['A-', 'A1', 'A2', 'B+', 'B1']


def make_pipeline(features):
    sub = df[features].dropna()
    imp = SimpleImputer(strategy='median')
    scl = StandardScaler()
    scl.fit(imp.fit_transform(sub))
    return imp, scl


imp1, scl1 = make_pipeline(FEAT_P1)
imp2, scl2 = make_pipeline(FEAT_P2)
imp3, scl3 = make_pipeline(FEAT_P3)

# Contexto distribución Caldas para Q1
MEAN_CALDAS = float(df['punt_global'].mean())
P10_CALDAS  = float(df['punt_global'].quantile(0.10))
P90_CALDAS  = float(df['punt_global'].quantile(0.90))
PMIN        = float(df['punt_global'].min())
PMAX        = float(df['punt_global'].max())

# Tasa base de bajo rendimiento en Caldas para Q2
BASE_RIESGO = float((df['punt_global'] < 230).mean() * 100)

# ─── 3. FUNCIONES DE PREDICCIÓN ──────────────────────────────────────────────

def predict_p1(vals):
    x = np.array(vals, dtype=float).reshape(1, -1)
    x = scl1.transform(imp1.transform(x))
    return float(model_p1.predict(x, verbose=0)[0][0])


def predict_p2(vals):
    x = np.array(vals, dtype=float).reshape(1, -1)
    x = scl2.transform(imp2.transform(x))
    return float(model_p2.predict(x, verbose=0)[0][0])


def predict_p3(vals):
    x = np.array(vals, dtype=float).reshape(1, -1)
    x = scl3.transform(imp3.transform(x))
    return model_p3.predict(x, verbose=0)[0]  # array de 5 probabilidades

# ─── 4. CONSTANTES DE ESTILO ─────────────────────────────────────────────────

CARD = {
    "background": "#ffffff",
    "border": "1px solid #e0e0e0",
    "borderRadius": "12px",
    "padding": "20px",
    "marginBottom": "16px",
    "boxShadow": "0 2px 4px rgba(0,0,0,0.06)",
}
LBL = {
    "fontWeight": "600", "fontSize": "0.82rem",
    "marginBottom": "4px", "display": "block", "color": "#444"
}
SEC = {
    "color": "#1a3a5c", "fontSize": "0.82rem", "fontWeight": "700",
    "textTransform": "uppercase", "letterSpacing": "0.5px",
    "marginTop": "18px", "marginBottom": "8px", "borderBottom": "1px solid #e8e8e8",
    "paddingBottom": "4px"
}

EDU_OPTS = [
    {"label": "Sin educación formal",       "value": 0},
    {"label": "Primaria",                   "value": 1},
    {"label": "Secundaria incompleta",      "value": 2},
    {"label": "Secundaria completa",        "value": 3},
    {"label": "Técnica o tecnológica",      "value": 4},
    {"label": "Profesional",                "value": 5},
    {"label": "Posgrado",                   "value": 6},
]

# ─── 5. LAYOUT ───────────────────────────────────────────────────────────────

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Saber 11 – Modelos Predictivos"

def lbl(text):
    return html.Span(text, style=LBL)

def sec(text):
    return html.P(text, style=SEC)

def radio(cid, opts, val):
    return dcc.RadioItems(id=cid, options=opts, value=val, inline=True,
                          style={"marginBottom": "10px", "fontSize": "0.85rem"})

def si_no(cid, val=0):
    return radio(cid, [{"label": " Sí", "value": 1}, {"label": " No", "value": 0}], val)


panel_inputs = html.Div([
    html.Div([
        html.H3("Perfil del estudiante",
                style={"marginTop": "0", "color": "#1a3a5c", "fontSize": "1rem"}),
        html.P("Complete el perfil para obtener las tres predicciones.",
               style={"fontSize": "0.78rem", "color": "#999",
                      "marginTop": "0", "marginBottom": "16px"}),

        sec("Información personal"),
        lbl("Género"),
        radio("genero", [{"label": " Femenino", "value": 0},
                         {"label": " Masculino", "value": 1}], 0),

        lbl("Estrato socioeconómico"),
        dcc.Slider(id="estrato", min=1, max=6, step=1, value=2,
                   marks={i: str(i) for i in range(1, 7)},
                   tooltip={"placement": "bottom", "always_visible": False}),
        html.Div(style={"marginBottom": "12px"}),

        lbl("Educación de la madre"),
        dcc.Dropdown(id="edu_madre", options=EDU_OPTS, value=3, clearable=False,
                     style={"marginBottom": "10px"}),

        lbl("Educación del padre"),
        dcc.Dropdown(id="edu_padre", options=EDU_OPTS, value=3, clearable=False,
                     style={"marginBottom": "10px"}),

        sec("Activos del hogar"),
        html.Div([
            html.Div([lbl("Computador"), si_no("computador")], style={"flex": "1"}),
            html.Div([lbl("Internet"),   si_no("internet")],   style={"flex": "1"}),
        ], style={"display": "flex", "gap": "10px"}),
        html.Div([
            html.Div([lbl("Automóvil"), si_no("automovil")], style={"flex": "1"}),
            html.Div([lbl("Lavadora"),  si_no("lavadora")],  style={"flex": "1"}),
        ], style={"display": "flex", "gap": "10px"}),

        sec("Institución educativa"),
        lbl("Tipo de colegio"),
        radio("privado",
              [{"label": " Oficial (público)", "value": 0},
               {"label": " Privado", "value": 1}], 0),

        lbl("Zona del colegio"),
        radio("urbano",
              [{"label": " Rural", "value": 0},
               {"label": " Urbana", "value": 1}], 1),

        lbl("¿Colegio bilingüe?"),
        si_no("bilingue"),

        sec("Puntajes académicos – solo para predicción de inglés"),
        html.P("Escala 0 – 100 por área.",
               style={"fontSize": "0.76rem", "color": "#aaa",
                      "marginTop": "-6px", "marginBottom": "8px"}),

        *[
            html.Div([
                lbl(nombre),
                dcc.Slider(id=sid, min=0, max=100, step=1, value=50,
                           marks={0: "0", 50: "50", 100: "100"},
                           tooltip={"placement": "bottom", "always_visible": False}),
                html.Div(style={"marginBottom": "10px"}),
            ])
            for nombre, sid in [
                ("Matemáticas",            "mate"),
                ("Lectura Crítica",        "lectura"),
                ("Ciencias Naturales",     "naturales"),
                ("Sociales y Ciudadanas",  "sociales"),
            ]
        ],

    ], style=CARD),
], style={"width": "310px", "flexShrink": "0"})


panel_predicciones = html.Div([

    # Q1
    html.Div([
        html.H3("P1 · Puntaje global esperado",
                style={"margin": "0 0 2px", "color": "#1a3a5c", "fontSize": "0.95rem"}),
        html.P("Regresión neuronal · Variable objetivo: punt_global (0–500)",
               style={"margin": "0 0 10px", "fontSize": "0.76rem", "color": "#999"}),
        dcc.Graph(id="fig_p1", config={"displayModeBar": False}),
        html.Div(id="txt_p1",
                 style={"fontSize": "0.84rem", "color": "#555",
                        "marginTop": "6px", "lineHeight": "1.6"}),
    ], style=CARD),

    # Q2
    html.Div([
        html.H3("P2 · Riesgo de bajo rendimiento",
                style={"margin": "0 0 2px", "color": "#1a3a5c", "fontSize": "0.95rem"}),
        html.P("Clasificación binaria · Umbral: puntaje global < 230 pts",
               style={"margin": "0 0 10px", "fontSize": "0.76rem", "color": "#999"}),
        dcc.Graph(id="fig_p2", config={"displayModeBar": False}),
        html.Div(id="txt_p2",
                 style={"fontSize": "0.84rem", "color": "#555",
                        "marginTop": "6px", "lineHeight": "1.6"}),
    ], style=CARD),

    # Q3
    html.Div([
        html.H3("P3 · Nivel de desempeño en inglés",
                style={"margin": "0 0 2px", "color": "#1a3a5c", "fontSize": "0.95rem"}),
        html.P("Clasificación multiclase · Niveles MCER: A−, A1, A2, B1, B+",
               style={"margin": "0 0 10px", "fontSize": "0.76rem", "color": "#999"}),
        dcc.Graph(id="fig_p3", config={"displayModeBar": False}),
        html.Div(id="txt_p3",
                 style={"fontSize": "0.84rem", "color": "#555",
                        "marginTop": "6px", "lineHeight": "1.6"}),
    ], style=CARD),

], style={"flex": "1", "minWidth": "0"})


app.layout = html.Div([

    # Header
    html.Div([
        html.H2("Saber 11 – Modelos Predictivos · Caldas",
                style={"margin": "0", "color": "#1a3a5c"}),
        html.P("Ministerio de Educación · Proyecto 2 ACTD · Grupo 6",
               style={"margin": "4px 0 0", "fontSize": "0.82rem", "color": "#888"}),
    ], style={
        "padding": "14px 24px",
        "borderBottom": "2px solid #dde3ec",
        "marginBottom": "20px",
        "background": "#ffffff",
    }),

    # Cuerpo: inputs | predicciones
    html.Div([
        panel_inputs,
        panel_predicciones,
    ], style={
        "display": "flex",
        "gap": "20px",
        "padding": "0 24px 24px",
        "alignItems": "flex-start",
    }),

], style={
    "fontFamily": "Arial, sans-serif",
    "maxWidth": "1280px",
    "margin": "0 auto",
    "background": "#f0f2f5",
    "minHeight": "100vh",
})

# ─── 6. CALLBACK ÚNICO ───────────────────────────────────────────────────────

@app.callback(
    Output("fig_p1", "figure"), Output("txt_p1", "children"),
    Output("fig_p2", "figure"), Output("txt_p2", "children"),
    Output("fig_p3", "figure"), Output("txt_p3", "children"),
    Input("estrato",   "value"), Input("edu_madre",  "value"),
    Input("edu_padre", "value"), Input("genero",     "value"),
    Input("computador","value"), Input("internet",   "value"),
    Input("automovil", "value"), Input("lavadora",   "value"),
    Input("privado",   "value"), Input("urbano",     "value"),
    Input("bilingue",  "value"),
    Input("mate",      "value"), Input("lectura",    "value"),
    Input("naturales", "value"), Input("sociales",   "value"),
)
def actualizar(estrato, edu_madre, edu_padre, genero,
               computador, internet, automovil, lavadora,
               privado, urbano, bilingue,
               mate, lectura, naturales, sociales):

    indice = (computador + internet + automovil + lavadora) / 4.0

    # ── Q1 ────────────────────────────────────────────────────────────────────
    vals = [estrato, edu_madre, edu_padre, indice,
            computador, internet, automovil, lavadora,
            privado, urbano, genero]

    pred = max(0.0, min(500.0, predict_p1(vals)))

    if pred >= MEAN_CALDAS:
        col1 = "#27ae60"
    elif pred >= P10_CALDAS:
        col1 = "#e67e22"
    else:
        col1 = "#c0392b"

    fig1 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        number={"font": {"size": 42, "color": col1}, "valueformat": ".1f"},
        gauge={
            "axis": {"range": [PMIN, PMAX], "tickwidth": 1},
            "bar":  {"color": col1},
            "steps": [
                {"range": [PMIN, P10_CALDAS],  "color": "#fdecea"},
                {"range": [P10_CALDAS, MEAN_CALDAS], "color": "#fff3e0"},
                {"range": [MEAN_CALDAS, P90_CALDAS], "color": "#e8f5e9"},
                {"range": [P90_CALDAS, PMAX],   "color": "#c8e6c9"},
            ],
            "threshold": {
                "line": {"color": "#1a3a5c", "width": 3},
                "thickness": 0.75, "value": MEAN_CALDAS,
            },
        },
    ))
    fig1.update_layout(height=210, margin=dict(l=20, r=20, t=10, b=10),
                       font=dict(family="Arial", size=11))

    diff = pred - MEAN_CALDAS
    sgn  = "+" if diff >= 0 else ""
    txt1 = (f"Puntaje predicho: {pred:.1f} pts  ·  "
            f"{sgn}{diff:.1f} pts vs promedio Caldas ({MEAN_CALDAS:.1f})")

    # ── Q2 ────────────────────────────────────────────────────────────────────
    prob = max(0.0, min(1.0, predict_p2(vals)))
    pct  = prob * 100

    if pct < 30:
        col2, lbl2 = "#27ae60", "Riesgo bajo"
    elif pct < 60:
        col2, lbl2 = "#e67e22", "Riesgo moderado"
    else:
        col2, lbl2 = "#c0392b", "Riesgo alto"

    fig2 = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%", "font": {"size": 42, "color": col2},
                "valueformat": ".1f"},
        delta={"reference": BASE_RIESGO, "valueformat": ".1f", "suffix": "%"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "ticksuffix": "%"},
            "bar":  {"color": col2},
            "steps": [
                {"range": [0, 30],  "color": "#e8f5e9"},
                {"range": [30, 60], "color": "#fff3e0"},
                {"range": [60, 100], "color": "#fdecea"},
            ],
            "threshold": {
                "line": {"color": "#555", "width": 2},
                "thickness": 0.75, "value": BASE_RIESGO,
            },
        },
    ))
    fig2.update_layout(height=210, margin=dict(l=20, r=20, t=10, b=10),
                       font=dict(family="Arial", size=11))

    txt2 = (f"{lbl2}  ·  Probabilidad de puntaje < 230: {pct:.1f}%  "
            f"(referencia Caldas: {BASE_RIESGO:.1f}%)")

    # ── Q3 ────────────────────────────────────────────────────────────────────
    vals3 = [mate, lectura, naturales, sociales,
             estrato, edu_madre, edu_padre, indice,
             computador, internet, automovil, lavadora,
             privado, urbano, bilingue, genero]

    probs3   = predict_p3(vals3)
    idx_pred = int(np.argmax(probs3))

    colors3 = ["#c8d6e5"] * len(CLASES_P3)
    colors3[idx_pred] = "#1a3a5c"

    fig3 = go.Figure(go.Bar(
        x=CLASES_P3,
        y=[p * 100 for p in probs3],
        marker_color=colors3,
        text=[f"{p * 100:.1f}%" for p in probs3],
        textposition="outside",
    ))
    fig3.update_layout(
        height=230,
        margin=dict(l=10, r=10, t=10, b=30),
        yaxis=dict(title="Probabilidad (%)", range=[0, 105]),
        xaxis=dict(title="Nivel MCER"),
        template="plotly_white",
        font=dict(family="Arial", size=12),
        plot_bgcolor="#ffffff",
    )

    txt3 = (f"Nivel predicho: {CLASES_P3[idx_pred]}  "
            f"(probabilidad: {probs3[idx_pred] * 100:.1f}%)")

    return fig1, txt1, fig2, txt2, fig3, txt3


# ─── 7. ENTRY POINT ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)

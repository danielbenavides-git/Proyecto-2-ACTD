import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from dash.exceptions import PreventUpdate
from dash import State
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import tensorflow as tf

# =======================
# 1) Cargar datos y modelos
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(BASE_DIR, "data", "saber11_features.csv"))

model_p1 = tf.keras.models.load_model(os.path.join(BASE_DIR, "models", "model_p1.keras"))
model_p2 = tf.keras.models.load_model(os.path.join(BASE_DIR, "models", "model_p2.keras"))
model_p3 = tf.keras.models.load_model(os.path.join(BASE_DIR, "models", "model_p3.keras"))

# Features por pregunta
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
CLASES_P3 = ['A-', 'A1', 'A2', 'B+', 'B1']

# Ajustar scalers sobre el dataset completo
def make_pipeline(features):
    sub = df[features].dropna()
    imp = SimpleImputer(strategy='median')
    scl = StandardScaler()
    scl.fit(imp.fit_transform(sub))
    return imp, scl

imp1, scl1 = make_pipeline(FEAT_P1)
imp2, scl2 = make_pipeline(FEAT_P2)
imp3, scl3 = make_pipeline(FEAT_P3)

# Estadísticas de contexto
MEAN_CALDAS = float(df['punt_global'].mean())
P10_CALDAS  = float(df['punt_global'].quantile(0.10))
P90_CALDAS  = float(df['punt_global'].quantile(0.90))
PMIN        = float(df['punt_global'].min())
PMAX        = float(df['punt_global'].max())
BASE_RIESGO = float((df['punt_global'] < 230).mean() * 100)

# Opciones compartidas de inputs
EDU_OPTS = [
    {"label": "Sin educación formal",  "value": 0},
    {"label": "Primaria",              "value": 1},
    {"label": "Secundaria incompleta", "value": 2},
    {"label": "Secundaria completa",   "value": 3},
    {"label": "Técnica o tecnológica", "value": 4},
    {"label": "Profesional",           "value": 5},
    {"label": "Posgrado",              "value": 6},
]

# =======================
# 2) Funciones de predicción
# =======================
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
    return model_p3.predict(x, verbose=0)[0]

# =======================
# 3) App
# =======================
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Saber 11 – Modelos Predictivos"


def fig_mensaje(titulo, mensaje):
    """Figura vacía con mensaje centrado (para evitar gráficos en blanco)."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        title=titulo,
        annotations=[dict(
            text=mensaje,
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=13, color="gray"),
            align="center"
        )],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=350,
        margin=dict(l=10, r=10, t=60, b=10),
        font=dict(family="Inter, Arial", size=12),
    )
    return fig


def panel_socioeconomico(prefix):
    """Panel de inputs socioeconómicos reutilizable por Tab 1 y Tab 2."""
    return html.Div([
        html.Div([
            html.Div([
                html.Label("Género"),
                dcc.RadioItems(
                    id=f"{prefix}_genero",
                    options=[{"label": "Femenino", "value": 0},
                             {"label": "Masculino", "value": 1}],
                    value=0, inline=True
                ),
            ], style={"flex": "1", "paddingRight": "10px"}),
            html.Div([
                html.Label("Estrato socioeconómico"),
                dcc.Slider(
                    id=f"{prefix}_estrato", min=1, max=6, step=1, value=2,
                    marks={i: str(i) for i in range(1, 7)},
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
            ], style={"flex": "2"}),
        ], style={"display": "flex", "marginBottom": "15px", "alignItems": "flex-end"}),

        html.Div([
            html.Div([
                html.Label("Educación de la madre"),
                dcc.Dropdown(id=f"{prefix}_edu_madre", options=EDU_OPTS,
                             value=3, clearable=False),
            ], style={"flex": "1", "paddingRight": "10px"}),
            html.Div([
                html.Label("Educación del padre"),
                dcc.Dropdown(id=f"{prefix}_edu_padre", options=EDU_OPTS,
                             value=3, clearable=False),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "marginBottom": "15px"}),

        html.Label("Activos del hogar"),
        html.Div([
            html.Div([
                html.Label("Computador", style={"fontSize": "0.85rem"}),
                dcc.RadioItems(id=f"{prefix}_computador", inline=True,
                    options=[{"label": " Sí", "value": 1}, {"label": " No", "value": 0}],
                    value=0),
            ], style={"flex": "1"}),
            html.Div([
                html.Label("Internet", style={"fontSize": "0.85rem"}),
                dcc.RadioItems(id=f"{prefix}_internet", inline=True,
                    options=[{"label": " Sí", "value": 1}, {"label": " No", "value": 0}],
                    value=0),
            ], style={"flex": "1"}),
            html.Div([
                html.Label("Automóvil", style={"fontSize": "0.85rem"}),
                dcc.RadioItems(id=f"{prefix}_automovil", inline=True,
                    options=[{"label": " Sí", "value": 1}, {"label": " No", "value": 0}],
                    value=0),
            ], style={"flex": "1"}),
            html.Div([
                html.Label("Lavadora", style={"fontSize": "0.85rem"}),
                dcc.RadioItems(id=f"{prefix}_lavadora", inline=True,
                    options=[{"label": " Sí", "value": 1}, {"label": " No", "value": 0}],
                    value=0),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "marginBottom": "15px"}),

        html.Div([
            html.Div([
                html.Label("Tipo de colegio"),
                dcc.RadioItems(id=f"{prefix}_privado", inline=True,
                    options=[{"label": "Oficial (público)", "value": 0},
                             {"label": "Privado", "value": 1}],
                    value=0),
            ], style={"flex": "1", "paddingRight": "10px"}),
            html.Div([
                html.Label("Zona del colegio"),
                dcc.RadioItems(id=f"{prefix}_urbano", inline=True,
                    options=[{"label": "Rural", "value": 0},
                             {"label": "Urbana", "value": 1}],
                    value=1),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "marginBottom": "10px"}),
    ])


app.layout = html.Div([
    html.Div([
        html.Img(
            src="/assets/logo-uniandes.png",
            style={"height": "60px", "marginRight": "15px"}
        ),
        html.H2(
            "Tablero Saber 11 – Modelos Predictivos (Caldas)",
            style={"margin": "0"}
        )
    ], style={
        "display": "flex",
        "alignItems": "center",
        "marginBottom": "15px",
        "borderBottom": "2px solid #e0e0e0",
        "paddingBottom": "10px"
    }),
    html.P("Ingresa el perfil de un estudiante para obtener predicciones de los tres modelos entrenados."),

    dcc.Tabs(id="tabs", value="tab1", children=[
        dcc.Tab(label="P1: Puntaje global esperado",      value="tab1"),
        dcc.Tab(label="P2: Riesgo de bajo rendimiento",   value="tab2"),
        dcc.Tab(label="P3: Nivel de desempeño en inglés", value="tab3"),
    ]),
    html.Div(id="contenido-tab"),
], style={"maxWidth": "1200px", "margin": "0 auto", "padding": "10px", "fontFamily": "Arial"})


# =======================
# 3) Layout Tab 1
# =======================
def layout_tab1():
    return html.Div([
        html.H3("P1. Puntaje global esperado (regresión)"),
        html.P(
            "Estima el puntaje global Saber 11 a partir del perfil socioeconómico "
            "e institucional del estudiante.",
            style={"color": "#555", "marginBottom": "15px"}
        ),

        panel_socioeconomico("p1"),

        html.Hr(style={"margin": "20px 0", "borderColor": "#e0e0e0"}),

        html.Div([
            html.Div([dcc.Graph(id="p1_gauge")],
                     style={"flex": "1", "paddingRight": "10px"}),
            html.Div([
                html.Div(id="p1_texto", style={
                    "fontSize": "0.9rem", "color": "#444",
                    "lineHeight": "1.7", "paddingTop": "30px"
                })
            ], style={"flex": "1"}),
        ], style={"display": "flex", "alignItems": "flex-start"}),
    ])


# layout Tab 2
def layout_tab2():
    return html.Div([
        html.H3("P2. Riesgo de bajo rendimiento (clasificación binaria)"),
        html.P(
            "Estima la probabilidad de que el estudiante obtenga un puntaje global "
            "inferior a 230 puntos (umbral de bajo rendimiento en Caldas).",
            style={"color": "#555", "marginBottom": "15px"}
        ),

        panel_socioeconomico("p2"),

        html.Hr(style={"margin": "20px 0", "borderColor": "#e0e0e0"}),

        html.Div([
            html.Div([dcc.Graph(id="p2_gauge")],
                     style={"flex": "1", "paddingRight": "10px"}),
            html.Div([
                html.Div(id="p2_texto", style={
                    "fontSize": "0.9rem", "color": "#444",
                    "lineHeight": "1.7", "paddingTop": "30px"
                })
            ], style={"flex": "1"}),
        ], style={"display": "flex", "alignItems": "flex-start"}),
    ])


# layout Tab 3
def layout_tab3():
    return html.Div([
        html.H3("P3. Nivel de desempeño en inglés (clasificación multiclase)"),
        html.P(
            "Predice el nivel MCER del estudiante en inglés: A−, A1, A2, B1, B+. "
            "Requiere además los puntajes en las otras áreas.",
            style={"color": "#555", "marginBottom": "15px"}
        ),

        panel_socioeconomico("p3"),

        html.Div([
            html.Label("¿Colegio bilingüe?"),
            dcc.RadioItems(
                id="p3_bilingue", inline=True,
                options=[{"label": " No", "value": 0}, {"label": " Sí", "value": 1}],
                value=0
            ),
        ], style={"marginBottom": "15px"}),

        html.Label("Puntajes en otras áreas (escala 0 – 100)"),
        html.Div([
            html.Div([
                html.Label("Matemáticas", style={"fontSize": "0.85rem"}),
                dcc.Slider(id="p3_mate", min=0, max=100, step=1, value=50,
                           marks={0: "0", 50: "50", 100: "100"},
                           tooltip={"placement": "bottom", "always_visible": False}),
            ], style={"flex": "1", "paddingRight": "15px"}),
            html.Div([
                html.Label("Lectura Crítica", style={"fontSize": "0.85rem"}),
                dcc.Slider(id="p3_lectura", min=0, max=100, step=1, value=50,
                           marks={0: "0", 50: "50", 100: "100"},
                           tooltip={"placement": "bottom", "always_visible": False}),
            ], style={"flex": "1", "paddingRight": "15px"}),
            html.Div([
                html.Label("Ciencias Naturales", style={"fontSize": "0.85rem"}),
                dcc.Slider(id="p3_naturales", min=0, max=100, step=1, value=50,
                           marks={0: "0", 50: "50", 100: "100"},
                           tooltip={"placement": "bottom", "always_visible": False}),
            ], style={"flex": "1", "paddingRight": "15px"}),
            html.Div([
                html.Label("Sociales y Ciudadanas", style={"fontSize": "0.85rem"}),
                dcc.Slider(id="p3_sociales", min=0, max=100, step=1, value=50,
                           marks={0: "0", 50: "50", 100: "100"},
                           tooltip={"placement": "bottom", "always_visible": False}),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "marginBottom": "20px"}),

        html.Hr(style={"margin": "20px 0", "borderColor": "#e0e0e0"}),

        html.Div([
            html.Div([dcc.Graph(id="p3_barras")], style={"flex": "2", "paddingRight": "10px"}),
            html.Div([
                html.H4("¿Qué muestra esta gráfica?",
                        style={"color": "#1a3a5c", "marginBottom": "10px"}),
                html.P(
                    "Cada barra representa la probabilidad de que el estudiante alcance "
                    "ese nivel MCER en inglés según su perfil. La barra resaltada en azul "
                    "oscuro es el nivel predicho.",
                    style={"fontSize": "0.85rem", "lineHeight": "1.6", "color": "#444"}
                ),
                html.Br(),
                html.Div(id="p3_texto", style={
                    "fontSize": "0.88rem", "color": "#1a3a5c",
                    "fontWeight": "600", "lineHeight": "1.6"
                }),
            ], style={
                "flex": "1",
                "paddingLeft": "24px",
                "paddingTop": "20px",
                "borderLeft": "3px solid #e0e0e0",
                "marginLeft": "10px",
            }),
        ], style={"display": "flex", "alignItems": "flex-start"}),
    ])


# =======================
# 4) Router Tabs
# =======================
@app.callback(Output("contenido-tab", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "tab1":
        return layout_tab1()
    if tab == "tab2":
        return layout_tab2()
    if tab == "tab3":
        return layout_tab3()


# =======================
# 5) Callbacks Tab 1
# =======================
@app.callback(
    Output("p1_gauge", "figure"),
    Output("p1_texto", "children"),
    Input("p1_estrato",    "value"),
    Input("p1_edu_madre",  "value"),
    Input("p1_edu_padre",  "value"),
    Input("p1_genero",     "value"),
    Input("p1_computador", "value"),
    Input("p1_internet",   "value"),
    Input("p1_automovil",  "value"),
    Input("p1_lavadora",   "value"),
    Input("p1_privado",    "value"),
    Input("p1_urbano",     "value"),
)
def actualizar_tab1(estrato, edu_madre, edu_padre, genero,
                    computador, internet, automovil, lavadora,
                    privado, urbano):
    indice = (computador + internet + automovil + lavadora) / 4.0
    vals   = [estrato, edu_madre, edu_padre, indice,
              computador, internet, automovil, lavadora,
              privado, urbano, genero]
    pred = max(PMIN, min(PMAX, predict_p1(vals)))

    if pred >= MEAN_CALDAS:
        color = "#27ae60"
    elif pred >= P10_CALDAS:
        color = "#e67e22"
    else:
        color = "#c0392b"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        number={"font": {"size": 48, "color": color}, "valueformat": ".1f"},
        gauge={
            "axis": {"range": [PMIN, PMAX], "tickwidth": 1},
            "bar":  {"color": color},
            "steps": [
                {"range": [PMIN, P10_CALDAS],       "color": "#fdecea"},
                {"range": [P10_CALDAS, MEAN_CALDAS], "color": "#fff3e0"},
                {"range": [MEAN_CALDAS, P90_CALDAS], "color": "#e8f5e9"},
                {"range": [P90_CALDAS, PMAX],        "color": "#c8e6c9"},
            ],
            "threshold": {
                "line": {"color": "#1a3a5c", "width": 3},
                "thickness": 0.75, "value": MEAN_CALDAS,
            },
        },
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=20, b=10),
        font=dict(family="Arial", size=12),
    )

    diff = pred - MEAN_CALDAS
    sgn  = "+" if diff >= 0 else ""
    texto = [
        html.P(f"Puntaje predicho: {pred:.1f} pts",
               style={"fontWeight": "700", "fontSize": "1.1rem", "color": color}),
        html.P(f"{sgn}{diff:.1f} pts respecto al promedio de Caldas ({MEAN_CALDAS:.1f} pts)."),
        html.P(
            "La línea azul en el indicador marca el promedio departamental. "
            "Las zonas de color indican los cuartiles de la distribución histórica.",
            style={"fontSize": "0.82rem", "color": "#888"}
        ),
    ]
    return fig, texto


# =======================
# 5b) Callback Tab 2
# =======================
@app.callback(
    Output("p2_gauge", "figure"),
    Output("p2_texto", "children"),
    Input("p2_estrato",    "value"),
    Input("p2_edu_madre",  "value"),
    Input("p2_edu_padre",  "value"),
    Input("p2_genero",     "value"),
    Input("p2_computador", "value"),
    Input("p2_internet",   "value"),
    Input("p2_automovil",  "value"),
    Input("p2_lavadora",   "value"),
    Input("p2_privado",    "value"),
    Input("p2_urbano",     "value"),
)
def actualizar_tab2(estrato, edu_madre, edu_padre, genero,
                    computador, internet, automovil, lavadora,
                    privado, urbano):
    indice = (computador + internet + automovil + lavadora) / 4.0
    vals   = [estrato, edu_madre, edu_padre, indice,
              computador, internet, automovil, lavadora,
              privado, urbano, genero]
    prob = max(0.0, min(1.0, predict_p2(vals)))
    pct  = prob * 100

    if pct < 30:
        color, etiqueta = "#27ae60", "Riesgo bajo"
    elif pct < 60:
        color, etiqueta = "#e67e22", "Riesgo moderado"
    else:
        color, etiqueta = "#c0392b", "Riesgo alto"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%", "font": {"size": 48, "color": color},
                "valueformat": ".1f"},
        delta={"reference": BASE_RIESGO, "valueformat": ".1f", "suffix": "%"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "ticksuffix": "%"},
            "bar":  {"color": color},
            "steps": [
                {"range": [0, 30],   "color": "#e8f5e9"},
                {"range": [30, 60],  "color": "#fff3e0"},
                {"range": [60, 100], "color": "#fdecea"},
            ],
            "threshold": {
                "line": {"color": "#555", "width": 2},
                "thickness": 0.75, "value": BASE_RIESGO,
            },
        },
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=20, b=10),
        font=dict(family="Arial", size=12),
    )

    texto = [
        html.P(f"{etiqueta} · {pct:.1f}% de probabilidad de bajo rendimiento",
               style={"fontWeight": "700", "fontSize": "1.1rem", "color": color}),
        html.P(
            f"La referencia (∆) es la tasa base de bajo rendimiento en Caldas: {BASE_RIESGO:.1f}%. "
            "Un valor positivo indica que este perfil tiene mayor riesgo que el promedio."
        ),
        html.P(
            "Bajo rendimiento se define como puntaje global < 230 pts, "
            "que corresponde al tercio inferior de la distribución en Caldas.",
            style={"fontSize": "0.82rem", "color": "#888"}
        ),
    ]
    return fig, texto


# =======================
# 5c) Callback Tab 3
# =======================
@app.callback(
    Output("p3_barras", "figure"),
    Output("p3_texto",  "children"),
    Input("p3_estrato",    "value"),
    Input("p3_edu_madre",  "value"),
    Input("p3_edu_padre",  "value"),
    Input("p3_genero",     "value"),
    Input("p3_computador", "value"),
    Input("p3_internet",   "value"),
    Input("p3_automovil",  "value"),
    Input("p3_lavadora",   "value"),
    Input("p3_privado",    "value"),
    Input("p3_urbano",     "value"),
    Input("p3_bilingue",   "value"),
    Input("p3_mate",       "value"),
    Input("p3_lectura",    "value"),
    Input("p3_naturales",  "value"),
    Input("p3_sociales",   "value"),
)
def actualizar_tab3(estrato, edu_madre, edu_padre, genero,
                    computador, internet, automovil, lavadora,
                    privado, urbano, bilingue,
                    mate, lectura, naturales, sociales):
    indice = (computador + internet + automovil + lavadora) / 4.0
    vals   = [mate, lectura, naturales, sociales,
              estrato, edu_madre, edu_padre, indice,
              computador, internet, automovil, lavadora,
              privado, urbano, bilingue, genero]
    probs    = predict_p3(vals)
    idx_pred = int(np.argmax(probs))

    colors = ["#c8d6e5"] * len(CLASES_P3)
    colors[idx_pred] = "#1a3a5c"

    fig = go.Figure(go.Bar(
        x=CLASES_P3,
        y=[p * 100 for p in probs],
        marker_color=colors,
        text=[f"{p * 100:.1f}%" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_white",
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Probabilidad (%)", range=[0, 105]),
        xaxis=dict(title="Nivel MCER"),
        font=dict(family="Arial", size=12),
        title=dict(x=0, xanchor="left"),
    )

    texto = [
        html.P(f"Nivel predicho: {CLASES_P3[idx_pred]}",
               style={"fontWeight": "700", "fontSize": "1.1rem", "color": "#1a3a5c"}),
        html.P(f"Probabilidad: {probs[idx_pred] * 100:.1f}%"),
    ]
    return fig, texto


if __name__ == "__main__":
    app.run(debug=True)
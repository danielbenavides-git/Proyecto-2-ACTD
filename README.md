# Modelos Predictivos Saber 11 - Caldas

**Curso:** Analítica Computacional para la Toma de Decisiones | **Profesor:** Juan F. Pérez | **GRUPO 6**

## Equipo
| Nombre | Código |
|---|---|
| Daniel Benavides | 202220428 |
| Juanita Cortés | 202222129 |
| Andrés Felipe Herrera | 202220888 |

## Descripción
Producto de analítica predictiva sobre los resultados de las pruebas Saber 11 en el departamento de Caldas, orientado al Ministerio de Educación como usuario final. El proyecto extiende el análisis exploratorio del Proyecto 1 mediante modelos de clasificación y regresión basados en redes neuronales, buscando responder tres preguntas de negocio: (1) ¿Cuál es el puntaje global esperado para un estudiante dado su perfil socioeconómico y el tipo de institución educativa? (regresión), (2) ¿Puede identificarse si un estudiante está en riesgo de obtener un puntaje por debajo del umbral de bajo desempeño? (clasificación binaria), y (3) ¿Puede predecirse el nivel de desempeño en inglés de un estudiante a partir de su perfil académico, socioeconómico y de género? (clasificación multiclase). Los modelos fueron rastreados con MLflow y el tablero interactivo permite al usuario interactuar con los modelos pre-entrenados.

## Estructura del repositorio
├── data/                        # Datos compartidos (saber11_features.csv)
├── tarea3/                      # Exploración de datos
│   └── data_analysis.ipynb
├── tarea4/                      # Modelamiento y experimentos MLflow
│   ├── ml_models.ipynb
│   └── keras_models/            # Modelos .keras generados durante entrenamiento
├── despliegue/                  # Tablero Dash + Dockerfile
│   ├── app.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── models/                  # Modelos serializados para producción
│       ├── model_p1.keras
│       ├── model_p2.keras
│       └── model_p3.keras
└── README.md

## Ejecución local
Instalar dependencias y correr el tablero desde la raíz del repositorio:
```bash
pip install -r despliegue/requirements.txt
python despliegue/app.py
```
Abrir en el navegador: `http://localhost:8050`

## Ejecución con Docker
```bash
docker build -t saber11-p2 ./despliegue
docker run -p 8050:8050 saber11-p2
```
Abrir en el navegador: `http://localhost:8050`

El tablero también está desplegado en **AWS EC2** usando Docker. URL: `http://32.197.254.106:8050`

## Datos
Los datos provienen del portal [Datos Abiertos Colombia](https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-11/kgxf-xxbe/data_preview) (actualización abril 2024). Se emplea el mismo subconjunto del Proyecto 1: registros del departamento de Caldas. El archivo `saber11_features.csv` contiene las variables de entrada procesadas para los tres modelos.

## Modelos
Los modelos son redes neuronales entrenadas con TensorFlow/Keras y se cargan pre-entrenados desde `despliegue/models/`:

| Archivo | Tipo | Variable objetivo |
|---|---|---|
| `model_p1.keras` | Regresión | `punt_global` (0–500) |
| `model_p2.keras` | Clasificación binaria | Riesgo de `punt_global` < 230 |
| `model_p3.keras` | Clasificación multiclase | Nivel de inglés: A−, A1, A2, B1, B+ |

El entrenamiento, los experimentos y las métricas de evaluación están documentados en **MLflow** (ver carpeta `tarea4/`).
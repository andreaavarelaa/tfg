#  Análisis de letras de canciones mediante inteligencia artificial generativa

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![NLP](https://img.shields.io/badge/NLP-spaCy%20%7C%20Transformers-green.svg)
![LLM](https://img.shields.io/badge/LLM-Llama%203.1-orange.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

Este repositorio contiene el código fuente y la investigación del **Trabajo de Fin de Grado** en Inteligencia Artificial. Consiste en un pipeline integral de *Data Engineering* e *Inteligencia Artificial* diseñado para extraer, procesar, analizar y comparar letras de canciones a nivel léxico y semántico.

---

## Descripción del proyecto

El sistema automatiza la recolección de letras de canciones mediante la API de Genius y aplica un enfoque de IA en múltiples capas:

1. **Ingeniería de datos y NLP clásico:** Limpieza de metadatos, lematización de textos mediante `spaCy` y extracción de métricas de complejidad léxica como el *Type-Token Ratio* y el índice *Flesch-Kincaid* usando `textstat`.
2. **Deep Learning:** Análisis de sentimiento dinámico a través de ventanas deslizantes y detección de emociones utilizando `RoBERTa`, además de clasificación temática *Zero-Shot* mediante `BART`.
3. **Machine Learning no supervisado:** Agrupación semántica de canciones implementando `BERTopic` con reducción de dimensionalidad matemática.
4. **IA Generativa:** Integración con la API de Groq. Tras realizar un riguroso *benchmark* de latencia y disponibilidad, se implementa **Llama 3.1 (8B)** para actuar como auditor de emociones y redactor de análisis literarios.

---

## Arquitectura

El proyecto ha sido diseñado bajo un paradigma de separación de responsabilidades para garantizar su escalabilidad en despliegues en la nube:

* **Backend (`main.py`):** Motor de procesamiento. Extrae los datos y ejecuta los Transformers, generando un Data Mart estático en formato `.csv`.
* **Frontend (`app.py`):** Interfaz web desarrollada en Streamlit. No ejecuta modelos de Machine Learning, sino que consume los resultados procesados, permitiendo un despliegue en la nube rápido y con bajo consumo de memoria RAM.

---

## Interfaz web

<!-- añadir capturas de pantalla de la interfaz web -->

<!-- ![imagen1](ruta_a_la_imagen_1.png) -->
<!-- *Descripción de la imagen 1* -->

---

## Estructura del repositorio

```text
├── data/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_Data_Collection.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 03b_Model_Testing.ipynb
│   ├── 04_Analysis_Visualization.ipynb
│   ├── 05_Comparison.ipynb
│   ├── 06_Topic_Modeling.ipynb
│   ├── 07_LLM_Summarization.ipynb
│   └── 07b_Benchmark.ipynb
├── main.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Instalación y uso local

1. Clonar el repositorio

```bash
git clone [https://github.com/](https://github.com/)[TU_USUARIO]/[NOMBRE_DEL_REPOSITORIO].git
cd [NOMBRE_DEL_REPOSITORIO]
```

2. Crear un entorno virtual e instalar dependencias

Se recomienda utilizar un entorno virtual para evitar conflictos. Las dependencias están optimizadas (incluyendo la versión CPU de PyTorch) para facilitar el despliegue.

```bash
python -m venv venv

venv\Scripts\activate       # para activar el entorno en Windows
source venv/bin/activate    # para activar el entorno en macOS/Linux

pip install -r requirements.txt
```

3. Configurar variables de entorno

Crea un archivo llamado `.env` en la raíz del proyecto y añade tus credenciales (necesarias solo para ejecutar `main.py`):

```bash
GENIUS_TOKEN="tu_token_de_genius"
GROQ_API_KEY="tu_api_key_de_groq"
EMAIL_USER="tu_correo_electronico"
EMAIL_PASS="tu_contraseña"
```

4. Ejecución del proyecto

Para interactuar con la interfaz web, ejecuta:

```bash
streamlit run app.py
```

Para procesar un nuevo artista hay dos opciones, directamente desde la interfaz web (desarrollar esto), o ejecutando el siguiente comando:

```bash
python main.py --artist "nombre del artista" --songs 300 --email "correo electrónico al que quieres que llegue el aviso cuando termine de procesarse"
```

## Créditos

**Desarrollado por:** Andrea Varela Fernández

**Titulación:** Grado en Inteligencia Artificial

**Universidad:** Universidade da Coruña

**Tutores:** Miguel Ángel Alonso Pardo y Jesús Vilares Ferro

**Año académico:** 2025/2026
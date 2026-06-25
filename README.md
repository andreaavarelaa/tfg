#  Análisis de letras de canciones mediante inteligencia artificial generativa

Este repositorio contiene el código fuente y la investigación del **Trabajo de Fin de Grado** en Inteligencia Artificial. Consiste en un sistema diseñado para extraer, procesar, analizar y comparar letras de canciones.

---

## Descripción del proyecto

El sistema automatiza la recolección de letras de canciones mediante la API de Genius y aplica un enfoque de IA en múltiples capas:

1. **Procesamiento del lenguaje natural:** Limpieza de metadatos, lematización de textos mediante `spaCy` y extracción de métricas de complejidad léxica como el *Type-Token Ratio* y el índice *Flesch-Kincaid* usando `textstat`.
2. **Aprendizaje profundo:** Análisis de sentimiento dinámico a través de ventanas deslizantes y detección de emociones utilizando `RoBERTa`, además de clasificación temática *Zero-Shot* mediante `BART`.
3. **Aprendizaje automático no supervisado:** Agrupación semántica de canciones implementando `BERTopic` con reducción de dimensionalidad matemática.
4. **IA Generativa:** Integración con la API de Groq, con la que se implementa **Llama 3.1 (8B)** para actuar como auditor de emociones y redactor de análisis literarios.

---

## Arquitectura

El proyecto ha sido diseñado bajo un paradigma de separación de responsabilidades para garantizar su escalabilidad en despliegues en la nube:

* **Backend (`main.py`):** Motor de procesamiento, extrae los datos y ejecuta los Transformers.
* **Frontend (`app.py`):** Interfaz web desarrollada en Streamlit, consume los resultados procesados y permite un despliegue en la nube rápido y con bajo consumo de memoria RAM.

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
git clone [https://github.com/](https://github.com/)andreaavarelaa/tfg.git
cd tfg
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

Para procesar un nuevo artista hay dos opciones, directamente desde la interfaz web, o ejecutando el siguiente comando:

```bash
python main.py --artist "nombre del artista" --songs 300 --email "correo electrónico al que quieres que llegue el aviso cuando termine de procesarse"
```

## Créditos

**Desarrollado por:** Andrea Varela Fernández

**Titulación:** Grado en Inteligencia Artificial

**Universidad:** Universidade da Coruña

**Tutores:** Miguel Ángel Alonso Pardo y Jesús Vilares Ferro

**Año académico:** 2025/2026

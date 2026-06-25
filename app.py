import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

import json
import math
import os
import subprocess
import sys
import requests
import re

custom_stopwords = list(ENGLISH_STOP_WORDS) + [
    'oh', 'yeah', 'cause', 'wanna', 'gonna', 'gotta', 'like', 'just', 'got', 'woah',
    'know', 'don', 'll', 've', 're', 'ain', 'ooh', 'ah', 'ha', 'la', 'eh', 'na', 'til',
    'im', 'm', 'did', 'didn', 'say', 'said', 'let', 'tell', 'way', 'make', 'wouldve',
    'uh', 'want', 'need', 'feel', 'think', 'right', 'time', 'thing', 'look', 'come',
    'dont', 'youre', 'ill', 'ive', 'hes', 'shes', 'were', 'theres', 'cant', 'wont',
    'didnt', 'id', 'youd', 'shouldve', 'isnt', 'youve', 'arent', 'wasnt', 'werent',
    'hadnt', 'doesnt', 'couldve', 'theyre', 'hey', 'thats', 'hasnt', 'isn', 'mm', 'won',
    'wasn', 'weren', 'doesn', 'hasn', 'hadn', 'couldn', 'wouldn', 'shouldn', 'aren',
    'somethin', 'nothin', 'bout', 'em', 'things', 'waitin', 'youll', 'theyll', 'ohohoh'
]

def get_top_n_words(corpus, n=None, ngram_range=(1, 1)):
    vec = CountVectorizer(stop_words=custom_stopwords, ngram_range=ngram_range).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

def validate_artist(name):
    try:
        url = f"https://itunes.apple.com/search"
        parameters = {
            "term": name,
            "entity": "musicArtist",
            "limit": 1
        }
        response = requests.get(url, params=parameters, timeout=5).json()

        if response['resultCount'] > 0:
            return response['results'][0]['artistName']
        else:
            return None
        
    except Exception as e:
        print(f"Error validando artista: {e}")
        return name

st.set_page_config(page_title="TFG: Análisis Musical", layout="wide", page_icon="🎵")

# st.title("Análisis lingüístico y emocional en letras de canciones")
# st.markdown("Exploración interactiva mediante NLP e IA generativa.")

@st.cache_data
def load_data(artist_name):
    filename_base = artist_name.lower().replace(" ", "_")

    ruta_final = f"data/processed/{filename_base}_final.csv"
    ruta_definitiva = f"data/processed/{filename_base}_definitivo3.csv"
    ruta_metrics = f"data/processed/{filename_base}_metrics.csv"

    if os.path.exists(ruta_final):
        df_temp = pd.read_csv(ruta_final)
    elif os.path.exists(ruta_definitiva):
        df_temp = pd.read_csv(ruta_definitiva)
    elif os.path.exists(ruta_metrics):
        df_temp = pd.read_csv(ruta_metrics)
    else:
        st.error(f"No se encontraron datos para {artist_name}")
        return pd.DataFrame()

    df_temp['artist'] = artist_name
    return df_temp

try:
    archivos = os.listdir("data/processed/")
except FileNotFoundError:
    archivos = []
    os.makedirs("data/processed/", exist_ok=True)

artistas_procesados = set()
for archivo in archivos:
    if archivo.endswith("_final.csv"):
        nombre = archivo.replace("_final.csv", "")
        artistas_procesados.add(nombre.replace("_", " ").title())
    elif archivo.endswith("_definitivo3.csv"):
        nombre = archivo.replace("_definitivo3.csv", "")
        artistas_procesados.add(nombre.replace("_", " ").title())
    elif archivo.endswith("_metrics.csv"):
        nombre = archivo.replace("_metrics.csv", "")
        artistas_procesados.add(nombre.replace("_", " ").title())

artists_available = sorted(list(artistas_procesados))

option = st.sidebar.selectbox(
    "Selecciona un artista", 
    artists_available + ["Solicitar nuevo artista..."]
)

if option == "Solicitar nuevo artista...":
    if 'artist_validated' not in st.session_state:
        st.session_state.artist_validated = None

    if 'previous_input' not in st.session_state:
        st.session_state.previous_input = ""

    st.header("Pipeline de IA automático")
    st.info("Introduce el nombre de un artista. Nuestro motor en segundo plano descargará sus letras, pasará los modelos de ML y generará los análisis literarios.")

    new_artist = st.text_input("Nombre del artista/banda (Ej: Taylor Swift, The Beatles):")

    if new_artist != st.session_state.previous_input:
        st.session_state.artist_validated = None
        st.session_state.previous_input = new_artist
    
    if not st.session_state.artist_validated:
        if st.button("Verificar artista:"):
            if new_artist:
                with st.spinner("Verificando.."):
                    oficial_artist = validate_artist(new_artist)

                if oficial_artist is None:
                    st.error(f"No hemos encontrado a '{new_artist}'. Por favor, revisa que esté bien escrito.")
                else:
                    st.session_state.artist_validated = oficial_artist
                    st.rerun()

            else:
                st.error("Por favor, introduce el nombre de un artista para iniciar el procesamiento.")

    if st.session_state.artist_validated:
        if st.session_state.artist_validated.lower() != new_artist.lower():
            st.info(f"Hemos corregido el nombre: **{st.session_state.artist_validated}**")
        else:
            st.success(f"Artista verificado: **{st.session_state.artist_validated}**")

        email = st.text_input("Introduce tu email (para notificarte cuando esté listo):")
        email_confirm = st.text_input("Confirma tu email:")
        max_canciones = st.slider("Número máximo de canciones a procesar (por defecto 300):", min_value=5, max_value=500, value=300, step=5)
        st.caption("Para una prueba rápida, no usar más de 20 canciones.")
    
        if st.button("Iniciar procesamiento"):
            if email != email_confirm:
                st.error("Los correos electrónicos no coinciden. Por favor, revísalos e inténtalo de nuevo.")
            else:
                st.success(f"¡Genial! Hemos puesto nuestros modelos a escuchar la discografía de **{st.session_state.artist_validated}**.")
                if email:
                    st.warning(f"Recibirás un email en {email} en cuanto el dashboard esté listo. Este proceso tardará un rato, dependiendo del tamaño de la discografía. Puedes seguir explorando otros artistas o cerrar la página. Cuando vuelvas, aparecerá en el menú lateral. Gracias por tu paciencia.")
                else:
                    st.warning("Este proceso tardará un rato, dependiendo del tamaño de la discografía. Puedes seguir explorando otros artistas o cerrar la página. Cuando vuelvas, aparecerá en el menú lateral. Gracias por tu paciencia.")

                comando = [sys.executable, "main.py", "--artist", st.session_state.artist_validated, "--songs", str(max_canciones)]
                if email.strip():
                    comando.extend(["--email", email])

                subprocess.Popen(comando)

                st.session_state.artist_validated = None
                st.session_state.previous_input = ""
else:
    df = load_data(option)

    if df.empty:
        st.error(f"El dataset de {option} está vacío o corrupto. Por favor, vuelve a procesarlo.")
    else:
        if 'album' in df.columns:
            df['album'] = df['album'].str.title()

        st.title(f"Análisis musical asistido por IA: {option}")
        st.markdown("Explora el universo lírico, métricas de complejidad y evolución emocional a través de modelos de Procesamiento de Lenguaje Natural.")

        # col1, col2, col3, col4 = st.columns(4)
        # col1.metric("Canciones", len(df))
        # col2.metric("Álbumes", df['album'].nunique())
        # col3.metric("Riqueza léxica media (TTR)", round(df['type_token_ratio'].mean(), 2))
        # col4.metric("Complejidad de lectura media (Flesch-Kincaid)", round(df['flesch_kincaid_grade'].mean(), 2))

        col1, col2, col3 = st.columns(3)
        col1.metric("Canciones", len(df), help="Número total de canciones procesadas para este artista.")
        col2.metric("Riqueza léxica media (TTR)", round(df['type_token_ratio'].mean(), 2), help="Mide la diversidad del vocabulario. Un valor más alto indica que el artista utiliza menos palabras repetidas.")
        col3.metric("Complejidad de lectura media (Flesch-Kincaid)", round(df['flesch_kincaid_grade'].mean(), 2), help="Mide la complejidad de lectura del texto. Un valor más alto indica mayor dificultad.")

        st.divider()

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Complejidad y evolución estilística", 
            "Temáticas y vocabulario",
            "Mapa emocional",
            "Arco narrativo",
            "Base de datos",
            "Comparativa"
        ])

        with tab1:
            st.header("Complejidad y evolución estilística")
            colA, colB = st.columns(2)

            with colA:
                st.subheader("Polaridad del sentimiento")
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                sns.histplot(data=df, x='sentiment_polarity', kde=True, ax=ax1, color='purple')
                ax1.set_title("Distribución de positividad/negatividad")
                st.pyplot(fig1, clear_figure=True)

            with colB:
                st.subheader("Riqueza léxica por álbum")
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                sns.boxplot(data=df, x='album', y='type_token_ratio', ax=ax2, palette='viridis', hue='album')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig2, clear_figure=True)  

            st.divider()
            colC, colD = st.columns(2)

            with colC:
                st.subheader("Estilo gramatical")
                fig3, ax3 = plt.subplots(figsize=(8, 5))
                sns.scatterplot(data=df, x='noun_ratio', y='verb_ratio', hue='album', palette='tab20', alpha=0.7, ax=ax3)
                ax3.set_title("Proporción de sustantivos vs verbos")
                ax3.set_xlabel("Ratio de sustantivos")
                ax3.set_ylabel("Ratio de verbos")
                ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                st.pyplot(fig3, clear_figure=True)
            
            with colD:
                st.subheader("Mención de entidades nombradas (NER)")
                fig4, ax4 = plt.subplots(figsize=(8, 5))
                sns.scatterplot(data=df, x='ent_person', y='ent_place', hue='album', palette='tab20', alpha=0.7, ax=ax4)
                ax4.set_title("Volumen de personas vs lugares por canción")
                ax4.set_xlabel("Personas mencionadas")
                ax4.set_ylabel("Lugares mencionados")
                ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                st.pyplot(fig4, clear_figure=True)

            st.divider()
            st.subheader("Evolución de la complejidad de lectura (Flesch-Kincaid)")
            st.markdown("Mide el nivel educativo necesario para comprender el texto. Valore más altos indican mayor complejidad.")

            fig13, ax13 = plt.subplots(figsize=(12, 4))
            sns.boxplot(data=df, x='album', y='flesch_kincaid_grade', ax=ax13, palette='magma', hue='album', legend=False)
            plt.xticks(rotation=45, ha='right')
            ax13.set_title("Dificultad de lectura por álbum")
            ax13.set_xlabel("")
            st.pyplot(fig13, clear_figure=True)

            st.divider()
            st.subheader("Evolución de la longitud de las canciones")
            fig17, ax17 = plt.subplots(figsize=(12, 5))
            sns.boxplot(data=df, x='album', y='word_count', ax=ax17, palette='muted', hue='album', legend=False)
            ax17.set_ylabel("Número de palabras")
            ax17.set_xlabel("")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig17, clear_figure=True)

        with tab2:
            st.header("Temáticas y vocabulario (Zero-Shot & WordClouds)")
            colE, colF = st.columns(2)

            with colE:
                st.subheader("Distribución de temáticas")

                if 'theme_zeroshot' in df.columns:
                    fig5, ax5 = plt.subplots(figsize=(8, 5))
                    sns.countplot(data=df, y='theme_zeroshot', order=df['theme_zeroshot'].value_counts().index, ax=ax5, palette="viridis", hue='theme_zeroshot', legend=False)
                    ax5.set_title("Temáticas dominantes (BART Zero-Shot)")
                    st.pyplot(fig5, clear_figure=True)
                else:
                    st.info("El análisis de temáticas zero-shot es exclusivo del caso de estudio principal (Taylor Swift).")
            
            with colF:
                st.subheader("Top 10 bi-gramas")
                clean_lyrics = df['lyrics_unique'].dropna().astype(str)
                top_bigrams = get_top_n_words(clean_lyrics, n=10, ngram_range=(2, 2))
                df_bigrams = pd.DataFrame(top_bigrams, columns=['bigram', 'count'])
                fig6, ax6 = plt.subplots(figsize=(8, 5))
                sns.barplot(data=df_bigrams, x='count', y='bigram', palette='Blues_r', ax=ax6, hue='bigram', legend=False)
                ax6.set_title("Pares de palabras más repetidos")
                st.pyplot(fig6, clear_figure=True)

            st.divider()
            st.subheader("Evolución temporal de las temáticas por álbum")
            
            theme_per_album = pd.crosstab(df['album'], df['theme_zeroshot'], normalize='index') * 100

            fig18, ax18 = plt.subplots(figsize=(14, 6))
            theme_per_album.plot(kind='bar', stacked=True, colormap='tab20', edgecolor='white', ax=ax18)
            ax18.set_ylabel("Porcentaje de canciones")
            ax18.set_xlabel("")
            plt.legend(title='Temática', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig18, clear_figure=True)

            if 'topic_name_llm' in df.columns:
                st.divider()
                st.subheader("Clústeres latentes")
                st.markdown("Agrupación matemática de canciones etiquetada semánticamente por IA generativa.")

                fig12, ax12 = plt.subplots(figsize=(10, 4))
                topic_counts = df['topic_name_llm'].value_counts()
                sns.barplot(x=topic_counts.values, y=topic_counts.index, palette='rocket', ax=ax12, hue=topic_counts.index, legend=False)
                
                ax12.set_title("Volumen de canciones por clúster temático")
                ax12.set_xlabel("ID del clúster (topic)")
                ax12.set_ylabel("Nº de canciones")
                st.pyplot(fig12, clear_figure=True)

                with st.expander("Explorar la composición de los clústeres"):
                    st.markdown("Palabras con mayor peso probabilístico (TF-IDF) dentro de cada cluster:")
                    topics = df[['topic_name_llm', 'topic_words']].drop_duplicates().dropna().reset_index(drop=True)

                    cols_exp = st.columns(2)
                    for i, row in topics.iterrows():
                        with cols_exp[i % 2]:
                            st.info(f"**{row['topic_name_llm']}**: {row['topic_words']}")
            else:
                st.info("Las temáticas avanzadas generadas por IA aún no están disponibles para este artista.")

            st.divider()
            st.subheader("Nubes de palabras por álbum")
            album_order = df['album'].unique()

            if len(album_order) > 0:
                cols = 2
                rows = math.ceil(len(album_order) / cols)
                rows = max(1, rows)

                fig7, ax7 = plt.subplots(rows, cols, figsize=(18, rows * 5))
                if not isinstance(ax7, np.ndarray):
                    ax7 = np.array([ax7])
                ax7 = ax7.flatten()

                for i, album in enumerate(album_order):
                    album_text = " ".join(df[df['album'] == album]['lyrics_unique'].dropna().astype(str))

                    if len(album_text.strip()) > 0:
                        wordcloud = WordCloud(
                            width=500, height=500,
                            background_color='white',
                            stopwords=custom_stopwords,
                            max_words=60,
                            colormap='inferno'
                        ).generate(album_text)

                        ax7[i].imshow(wordcloud, interpolation='bilinear')
                    
                    ax7[i].set_title(album, fontsize=18, fontweight='bold', pad=15)
                    ax7[i].axis('off')
                
                for j in range(i + 1, len(ax7)):
                    fig7.delaxes(ax7[j])

                plt.tight_layout()
                st.pyplot(fig7, clear_figure=True)
            
            else:
                st.info("No hay datos de álbumes suficientes para generar las nubes de palabras.")

        with tab3:
            st.header("Mapa emocional")
            colG, colH = st.columns([1, 2])

            with colG:
                st.subheader("Emoción dominante global")
                fig8, ax8 = plt.subplots(figsize=(7, 6))
                emotion_count = df['emotion_goemotions'].value_counts()
                colors_pie = sns.color_palette("pastel")[0:len(emotion_count)]

                porcentajes = 100 * emotion_count / emotion_count.sum()
                legend_labels = [f"{emocion} ({pct:.1f}%)" for emocion, pct in zip(emotion_count.index.str.title(), porcentajes)]
                
                wedges, texts = ax8.pie(emotion_count, colors=colors_pie, startangle=140)
                ax8.legend(wedges, legend_labels, title="Emociones", loc='center left', bbox_to_anchor=(0.9, 0.5))
                st.pyplot(fig8, clear_figure=True)

            with colH:
                st.subheader("Mapa de calor por álbum")
                emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
                album_order = df['album'].unique()
                album_emotions = df.groupby('album')[emotions].mean().reindex(album_order)

                fig9, ax9 = plt.subplots(figsize=(10, 6))
                sns.heatmap(album_emotions.T, cmap='YlGnBu', annot=True, fmt=".2f", ax=ax9)

                # ax9.set_title(f"Intensidad media de emociones por álbum: {artist}", fontsize=16, fontweight='bold', pad=15)
                ax9.set_xlabel("Álbum")
                ax9.set_ylabel("Emoción")
                st.pyplot(fig9, clear_figure=True)

        with tab4:
            st.header("Arco narrativo: análisis profundo de una canción")
            st.markdown("Selecciona una canción para ver cómo evoluciona su riqueza léxica y emoción a lo largo de su estructura.")

            song = st.selectbox("Selecciona una canción:", df['title'].unique())
            song_data = df[df['title'] == song].iloc[0]

            with st.expander("Ver texto crudo vs normalizado"):
                colK, colL = st.columns(2)

                with colK:
                    st.markdown("**Texto original:**")
                    st.info(str(song_data.get('lyrics_full', ''))[:500] + "...")
                
                with colL:
                    st.markdown("**Texto lematizado y sin stopwords:**")
                    st.warning(str(song_data.get('lyrics_norm', ''))[:500] + "...")

            st.write(f"**Álbum:** {song_data['album']} | **Emoción dominante:** {song_data['dominant_emotion'].title()}")

            colors = {"admiration": "#9d4edd", "amusement": "#ff9f1c", "anger": "#d00000", 
                "annoyance": "#9d0208", "approval": "#06d6a0", "caring": "#ffb5a7", 
                "confusion": "#5a189a", "curiosity": "#00b4d8", "desire": "#9e2a2b", 
                "disappointment": "#5c677d", "disapproval": "#6a994e", "disgust": "#386641", 
                "embarrassment": "#e5989b", "excitement": "#ff6b35", "fear": "#2b2d42", 
                "gratitude": "#48cae4", "grief": "#14213d", "joy": "#ffd700", "love": "#f72585", 
                "nervousness": "#8d99ae", "optimism": "#8cb369", "pride": "#4361ee", 
                "realization": "#0077b6", "relief": "#ade8f4", "remorse": "#7f4f24", 
                "sadness": "#03045e", "surprise": "#00f5d4", "neutral": "#ced4da"}

            if 'ttr_genius_seq' in df.columns:
                try:
                    seq_emociones = song_data['emotion_genius_seq']
                    seq_ttr = song_data['ttr_genius_seq']

                    if isinstance(seq_emociones, str) and isinstance(seq_ttr, str):
                        emotion_seq = json.loads(song_data['emotion_genius_seq'])
                        ttr_seq = json.loads(song_data['ttr_genius_seq'])

                        x_labels = [f"Bloque {i+1}" for i in range(len(ttr_seq))]

                        fig10, ax10 = plt.subplots(figsize=(14, 6))
                        ax10.plot(x_labels, ttr_seq, color='gray', linestyle='--', zorder=1, alpha=0.5)

                        seen_emotions = set()
                        for i in range(len(x_labels)):
                            emotion = emotion_seq[i]
                            color = colors.get(emotion, "#ced4da")
                            label = emotion if emotion not in seen_emotions else ""
                            ax10.scatter(x_labels[i], ttr_seq[i], color=color, s=200, zorder=2, label=label, edgecolor='black')
                            seen_emotions.add(emotion)
                        
                        ax10.set_title(f"Arco narrativo de '{song}': Riqueza léxica (TTR) y evolución emocional", fontsize=16, fontweight='bold', pad=15)
                        ax10.set_xlabel("Estructura de la canción", fontsize=12)
                        ax10.set_ylabel("Riqueza léxica", fontsize=12)

                        ax10.tick_params(axis='x', rotation=45)
                        ax10.grid(True, linestyle=':', alpha=0.7)
                        ax10.legend(title="Emoción del bloque", bbox_to_anchor=(1.05, 1), loc='upper left')

                        fig10.tight_layout()
                        st.pyplot(fig10, clear_figure=True)
                    
                    else:
                        st.warning("Estructura no detectada: No hay datos de secuencias para esta canción.")
                
                except Exception as e:
                    st.warning("Estructura no detectada: Esta canción no tiene corchetes de Genius (ej. [Chorus]) para generar el arco narrativo.")
            
            else:
                st.info("El análisis literario profundo no está disponible para este artista.")


            st.subheader("Análisis semántico por IA generativa")

            # if 'llm_meaning_4' in df.columns:
            #     st.write("**Compara el razonamiento de varios modelos LLM:**")

            #     models = {
            #         "llama-3.1-8b-instant": ("llm_meaning", "llm_validation"),
            #         "moonshotai/kimi-k2-instruct": ("llm_meaning_3", "llm_validation_3"),
            #         "meta-llama/llama-4-scout-17b-16e-instruct": ("llm_meaning_4", "llm_validation_4")
            #     }

            #     selected_model = st.radio(
            #         "Selecciona el motor de inferencia:",
            #         list(models.keys()),
            #         horizontal=True
            #     )

            #     col_meaning, col_validation = models[selected_model]

            #     colI, colJ = st.columns(2)

            #     with colI:
            #         st.info("**Significado de la letra:**")
            #         st.write(song_data[col_meaning])

            #     with colJ:
            #         st.success("**Validación de la emoción (Zero-Shot audit):**")
            #         st.write(song_data[col_validation])

            #     st.divider()
            #     st.subheader("Rendimiento de inferencia")
            #     st.markdown("Comparativa de latencia media de los modelos al procesar la discografía mediante la API de Groq.")

            #     try:
            #         df_benchmark = pd.read_csv("data/processed/benchmark_results.csv")

            #         times = df_benchmark.groupby('Modelo')['Tiempo (s)'].mean().reset_index()
            #         times = times.sort_values(by='Tiempo (s)', ascending=True)

            #         fig14, ax14 = plt.subplots(figsize=(8, 3))
            #         sns.barplot(data=times, x='Tiempo (s)', y='Modelo', palette='crest', ax=ax14)
            #         ax14.set_title("Latencia media por canción")
            #         ax14.set_xlabel("Segundos")
            #         ax14.set_ylabel("")
            #         st.pyplot(fig14, clear_figure=True)
                
            #     except FileNotFoundError:
            #         st.warning("El archivo de resultados del benchmark no se encuentra en la ruta especificada.")
            
            # elif 'llm_meaning' in df.columns:
            if 'llm_meaning' in df.columns:
                colI, colJ = st.columns(2)

                with colI:
                    st.info("**Significado de la letra:**")
                    st.write(song_data.get('llm_meaning', 'Sin descripción') if pd.notna(song_data.get('llm_meaning')) else 'Análisis no disponible.')

                with colJ:
                    st.success("**Validación de la emoción (Zero-Shot audit):**")
                    st.write(song_data.get('llm_validation', 'Sin validación') if pd.notna(song_data.get('llm_validation')) else 'Validación no disponible.')

                st.caption("*Nota: Este análisis ha sido generado por Inteligencia Artificial y, aunque ha sido validado, podría contener ligeras imprecisiones literarias.*")
                            
            else:
                st.info("Los resúmenes interpretativos mediante IA generativa no están disponibles para este artista.")

        with tab5:
            st.header("Base de datos")
            st.markdown("Explora las métricas exactas de cada canción.")

            columns = ['title', 'album', 'year', 'type_token_ratio', 'flesch_kincaid_grade', 'sentiment_polarity', 'dominant_emotion']
            st.dataframe(df[columns], use_container_width=True)

        with tab6:
            st.header("Comparativa de dos artistas")

            if len(artists_available) < 2:
                st.info("Necesitas procesar al menos dos artistas para acceder a la comparativa.")
            else:
                st.markdown("Selecciona dos artistas para analizar sus diferencias y similitudes.")

                colM, colN = st.columns(2)
                with colM:
                    artist1 = st.selectbox("Selecciona el primer artista:", artists_available, index=0)
                with colN:
                    artist2 = st.selectbox("Selecciona el segundo artista:", artists_available, index=1)

                if artist1 == artist2:
                    st.warning("Por favor, selecciona dos artistas diferentes para realizar la comparativa.")
                else:
                    df1 = load_data(artist1)
                    df2 = load_data(artist2)

                    df_combined = pd.concat([df1, df2], ignore_index=True)

                    st.divider()

                    colO, colP = st.columns(2)

                    with colO:
                        st.subheader("Distribución del sentimiento")
                        fig11, ax11 = plt.subplots(figsize=(8, 5))
                        sns.kdeplot(data=df_combined, x='sentiment_polarity', hue='artist', fill=True, palette='Set1', ax=ax11)
                        ax11.set_title("Polaridad (TextBlob)")
                        ax11.set_xlabel("Negatividad <---> Positividad")
                        st.pyplot(fig11, clear_figure=True)
                
                    with colP:
                        st.subheader("Riqueza léxica (TTR)")
                        fig15, ax15 = plt.subplots(figsize=(8, 5))
                        sns.kdeplot(data=df_combined, x='type_token_ratio', hue='artist', fill=True, palette='Set2', ax=ax15)
                        ax15.set_title("Diversidad de vocabulario")
                        ax15.set_xlabel("Type-Token Ratio")
                        st.pyplot(fig15, clear_figure=True)

                    st.divider()

                    st.subheader("Comparativa de emociones dominantes")
                    fig16, ax16 = plt.subplots(figsize=(10, 5))

                    top_emotions = df_combined['emotion_goemotions'].value_counts().head(8).index
                    df_filtered = df_combined[df_combined['emotion_goemotions'].isin(top_emotions)]

                    sns.countplot(data=df_filtered, x='emotion_goemotions', hue='artist', palette='Set3', ax=ax16)
                    ax16.set_title("Emociones más frecuentes por artista")
                    ax16.set_xlabel("Emoción")
                    ax16.set_ylabel("Número de canciones")
                    plt.xticks(rotation=45)
                    st.pyplot(fig16, clear_figure=True)
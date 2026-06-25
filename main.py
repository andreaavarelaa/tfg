import os
import time
import lyricsgenius
import re
import spacy
import textstat
import json
import argparse
import gc
import smtplib

import pandas as pd

from dotenv import load_dotenv
from textblob import TextBlob
from transformers import pipeline
from collections import Counter
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from groq import Groq
from email.message import EmailMessage

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer

def step_1_data_collection(artist_name, max_songs):
    print(f"\nPaso 1: Descargando top {max_songs} canciones de '{artist_name}'...")

    load_dotenv()
    GENIUS_TOKEN = os.getenv("GENIUS_TOKEN")

    if not GENIUS_TOKEN:
        raise ValueError("ERROR: No se encontró el token de Genius en el archivo .env")
    
    genius = lyricsgenius.Genius(GENIUS_TOKEN, timeout=60, retries=5)
    genius.verbose = True
    genius.remove_section_headers = False
    genius.skip_non_songs = True

    try:
        artist_data = genius.search_artist(artist_name, max_songs=max_songs, sort="popularity")
    except Exception as e:
        raise ConnectionError(f"Error al conectar con Genius: {e}")
    
    if artist_data is None:
        raise ValueError(f"No se encontró el artista '{artist_name}' en Genius.")
    
    all_songs = []
    for song in artist_data.songs:
        song_dict = song.to_dict()

        release_components = song_dict.get('release_date_components')
        song_year = release_components.get('year') if release_components else None

        album_data = song_dict.get('album')
        album_name = album_data.get('name') if album_data else "Desconocido"
        
        all_songs.append({
            "artist": artist_name,
            "title": song.title,
            "album": album_name,
            "year": song_year,
            "lyrics": song.lyrics
        })
        time.sleep(0.1)

    df = pd.DataFrame(all_songs)

    df_clean = df.drop_duplicates(subset=['title'], keep='first')
    df_clean = df_clean.sort_values(by=['year', 'album'])

    print(f"Paso 1 completado: {len(df_clean)} canciones obtenidas.")

    filename = artist_name.lower().replace(" ", "_")
    os.makedirs("data/interim", exist_ok=True)
    raw_path = f"data/interim/{filename}_raw.csv"
    df_clean.to_csv(raw_path, index=False)

    return df_clean

def step_2_preprocessing(df, artist_name):
    print(f"\nPaso 2: Limpiando y preprocesando {len(df)} canciones...")

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Descargando modelo de spaCy 'en_core_web_sm'...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    blacklist = ["Remix", "Version", "Acoustic", "Prologue", "Live", "Demo", "Mix", "Edit", "Instrumental", "Karaoke", "Memo", "Magazine"]

    whitelists = {
        "taylor swift": [("taylor's version", "taylors_regrabacion"), ("taylors version", "taylors_regrabacion")],
    }

    def is_noise(title):
        title_lower = str(title).lower().replace("’", "'")
        title_safe = title_lower

        if artist_name.lower() in whitelists:
            for original_title, new_title in whitelists[artist_name.lower()]:
                title_safe = title_safe.replace(original_title, new_title)

        for keyword in blacklist:
            if keyword.lower() in title_safe:
                return True
            
        return False
    
    mask = df['title'].apply(is_noise)
    dropped_count = mask.sum()
    df_clean = df[~mask].copy()
    print(f"Eliminadas {dropped_count} canciones consideradas ruido.")

    if len(df_clean) == 0:
        raise ValueError(f"Todas las canciones obtenidas eran ruido, instrumentales o remixes. No se puede analizar a {artist_name}.")

    def clean_lyrics(text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
    
    df_clean['lyrics_full'] = df_clean['lyrics'].apply(clean_lyrics)

    empty_songs = df_clean['lyrics_full'].str.strip() == ""
    dropped_empty = empty_songs.sum()
    df_clean = df_clean[~empty_songs].copy()
    print(f"Eliminadas {dropped_empty} canciones por no tener letra.")

    if len(df_clean) == 0:
        raise ValueError("No quedaron canciones con letra válida tras la limpieza.")

    def get_unique_lines(text):
        if not isinstance(text, str):
            return ""
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = re.sub(r'\b(\w+)(?:[\s,]+\1\b)+', r'\1', line, flags=re.IGNORECASE)
            line = re.sub(r'(\b.+?)(?:,\s+\1)+', r'\1', line, flags=re.IGNORECASE)
            cleaned_lines.append(line)
        unique_lines = list(dict.fromkeys([line for line in cleaned_lines if line]))
        return "\n".join(unique_lines)
    
    df_clean['lyrics_unique'] = df_clean['lyrics_full'].apply(get_unique_lines)

    def lemmatize_text(text):
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
        return " ".join(tokens)
    
    print("Lematizando las letras de las canciones...")
    df_clean['lyrics_norm'] = df_clean['lyrics_unique'].apply(lemmatize_text)

    filename = artist_name.lower().replace(" ", "_")
    processed_path = f"data/interim/{filename}_processed.csv"
    df_clean.to_csv(processed_path, index=False)

    print(f"Paso 2 completado: {len(df_clean)} canciones limpias y normalizadas guardadas.")

    df_clean = df_clean.reset_index(drop=True)
    return df_clean

def step_3_feature_engineering(df, artist_name):
    print(f"\nPaso 3: Extracción de métricas NLP e IA para '{artist_name}'...")
    nlp = spacy.load("en_core_web_sm")

    print("Calculando métricas léxicas...")
    def calculate_lexical_metrics(row):
        text_full = str(row.get('lyrics_full', ''))
        text_unique = str(row.get('lyrics_unique', ''))

        words = text_full.split()
        word_count = len(words)
        line_count = len(text_full.split('\n'))

        unique_words = text_unique.lower().split()
        vocab_size = len(set(unique_words))
        ttr = vocab_size / len(unique_words) if len(unique_words) > 0 else 0

        try:
            readability = textstat.flesch_kincaid_grade(text_full)
        except:
            readability = 0

        doc = nlp(text_full)
        stopwords_count = sum([1 for token in doc if token.is_stop])
        stopwords_ratio = stopwords_count / word_count if word_count > 0 else 0

        return pd.Series([word_count, line_count, vocab_size, ttr, readability, stopwords_ratio])
    
    df[['word_count', 'line_count', 'vocab_size', 'type_token_ratio', 'flesch_kincaid_grade', 'stopwords_ratio']] = df.apply(calculate_lexical_metrics, axis=1)

    print("Análisis sintáctico y entidades nombradas...")
    def grammar_analysis(text):
        doc = nlp(str(text))
        pos_counts = Counter([token.pos_ for token in doc])
        total_tokens = len(doc)

        noun_ratio = pos_counts['NOUN'] / total_tokens if total_tokens > 0 else 0
        verb_ratio = pos_counts['VERB'] / total_tokens if total_tokens > 0 else 0
        adj_ratio = pos_counts['ADJ'] / total_tokens if total_tokens > 0 else 0
        pron_ratio = pos_counts['PRON'] / total_tokens if total_tokens > 0 else 0

        ents = Counter([ent.label_ for ent in doc.ents])
        return pd.Series([noun_ratio, verb_ratio, adj_ratio, pron_ratio, ents['PERSON'], ents['GPE'] + ents['LOC'], ents['ORG']])

    df[['noun_ratio', 'verb_ratio', 'adj_ratio', 'pron_ratio', 'ent_person', 'ent_place', 'ent_org']] = df['lyrics_full'].apply(grammar_analysis)

    df['sentiment_polarity'] = df['lyrics_full'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

    print("Detectando 7 emociones dominantes...")
    classifier_emotions = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, truncation=True)

    emotions_list = []
    for lyrics in df['lyrics_full']:
        preds = classifier_emotions(str(lyrics))
        scores = {item['label']: item['score'] for item in preds[0]}
        emotions_list.append(scores)
    
    emotions_df = pd.DataFrame(emotions_list)
    df = pd.concat([df, emotions_df], axis=1)
    df['dominant_emotion'] = emotions_df.idxmax(axis=1)

    print("Clasificación Zero-Shot de narrativas...")
    clf_zeroshot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    labels = ["love", "heartbreak", "empowerment", "nostalgia", "friendship", "self-reflection", "growth", "fun", "sadness", "anger", 
              "revenge", "anxiety", "hope", "loneliness", "celebration", "loss", "forgiveness"]

    themes_zs, scores_zs = [], []
    for lyrics in df['lyrics_full']:
        clean_lyrics = str(lyrics).replace('\n', ' ')

        if len(clean_lyrics) < 5:
            themes_zs.append("Sin tema")
            scores_zs.append(0.0)
        else:
            res_zs = clf_zeroshot(clean_lyrics, candidate_labels=labels)
            themes_zs.append(res_zs['labels'][0])
            scores_zs.append(res_zs['scores'][0])
    
    df['theme_zeroshot'] = themes_zs
    df['theme_score'] = scores_zs

    del clf_zeroshot
    gc.collect()

    print("Mapeo detallado de 28 emociones...")
    clf_goemotions = pipeline("text-classification", model="samlowe/roberta-base-go_emotions", top_k=None, truncation=True)

    emotions_go = []
    for lyrics in df['lyrics_full']:
        res_go = clf_goemotions(str(lyrics))[0]
        top_emo = res_go[0] if isinstance(res_go, list) else res_go
        emotions_go.append(top_emo['label'])
    df['emotion_goemotions'] = emotions_go

    print("Construyendo arcos narrativos estructurales...")
    def get_genius_sections(text):
        sections = re.split(r'\[.*?\]', str(text))
        return [sec.strip() for sec in sections if len(sec.strip()) > 5]
    
    def calculate_ttr(text):
        words = str(text).lower().split()
        return len(set(words)) / len(words) if words else 0.0
    
    emotion_genius_seq, ttr_genius_seq = [], []
    for lyrics_raw in df['lyrics']:
        chunks = get_genius_sections(lyrics_raw)
        song_emo_seq, song_ttr_seq = [], []

        for chunk in chunks:
            res_go = clf_goemotions(chunk)[0]
            top_emo = res_go[0]['label'] if isinstance(res_go, list) else res_go['label']
            song_emo_seq.append(top_emo)
            song_ttr_seq.append(round(calculate_ttr(chunk), 3))

        emotion_genius_seq.append(json.dumps(song_emo_seq))
        ttr_genius_seq.append(json.dumps(song_ttr_seq))
    
    df['emotion_genius_seq'] = emotion_genius_seq
    df['ttr_genius_seq'] = ttr_genius_seq

    filename = artist_name.lower().replace(" ", "_")
    processed_path = f"data/interim/{filename}_metrics.csv"
    df.to_csv(processed_path, index=False)

    print(f"Paso 3 completado. Guardado en {processed_path}.")
    return df

def step_4_topic_modeling(df, artist_name):
    print(f"\nPaso 4: Topic Modeling y clustering con Llama 3...")

    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("ERROR: No se encontró la clave de API de Groq en el archivo .env")
    
    client = Groq(api_key=GROQ_API_KEY)

    print("Generando embeddings de las canciones...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    lyrics_list = df['lyrics_norm'].fillna("").tolist()
    embeddings = embedding_model.encode(lyrics_list, show_progress_bar=False)

    n_songs = len(df)
    words_dict = {}
    global_context = ""

    if n_songs < 15:
        df['topic_bertopic'] = 0

        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(lyrics_list)
        word_counts = X.sum(axis=0)
        words_freq = [(word, word_counts[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        top_words = ", ".join([w[0] for w in words_freq[:30]])

        words_dict[0] = top_words
        global_context += f"- Clúster 0: {top_words}\n"
    else:
        print("Buscando el número óptimo de clústeres con silhouette score...")
        max_k = min(10, max(4, n_songs // 15))
        k_values = [k for k in range(4, max_k + 2) if k < n_songs]
        if not k_values:
            k_values = [2, 3]
    
        best_score = -1
        best_k = k_values[0]
        best_model = None

        vectorizer_model = CountVectorizer(stop_words='english')

        for k in k_values:
            cluster_model = KMeans(n_clusters=k, random_state=42)
            topic_model = BERTopic(language="english", vectorizer_model=vectorizer_model, hdbscan_model=cluster_model)
            topics, _ = topic_model.fit_transform(lyrics_list, embeddings)

            try:
                score = silhouette_score(embeddings, topics)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_model = topic_model
            except Exception:
                pass

        if best_model is None:
            best_model = topic_model
            best_k = k_values[-1]

        print(f"Óptimo matemático encontrado: {best_k} clústeres (Score: {round(best_score, 3)}).")

        df['topic_bertopic'] = best_model.topics_

        print("Extrayendo palabras clave y generando contexto global...")
        info_topics = best_model.get_topic_info()

        for _, row in info_topics.iterrows():
            topic_id = row['Topic']
            if topic_id != -1:
                words_topic = best_model.get_topic(topic_id)
                top_words = ", ".join([word[0] for word in words_topic[:30]])
                words_dict[topic_id] = top_words
                global_context += f"- Clúster {topic_id}: {top_words}\n"
    
    print("Nombrando los clústeres...")
    names = {}
    already_used_names = []

    for topic_id, top_words in words_dict.items():
        used = ", ".join(already_used_names) if already_used_names else "Ninguno todavía"

        prompt = f"""
        You are a highly analytical music data scientist creating strict, conceptual taxonomy tags for {artist_name}'s discography.
        I have grouped her songs into several clusters. Here is the FULL LANDSCAPE of all clusters' words to prevent overlap:
        
        {global_context}

        TAGS ALREADY ASSIGNED TO OTHER CLUSTERS (DO NOT REPEAT THESE):
        [{used}]

        Keywords for cluster {topic_id}:
        [{top_words}]

        Your task: Categorize ONLY "Cluster {topic_id}" into a broad, recognizable "Song Theme" or "Musical Trope" in Spanish.

        CRITICAL RULES:
        1. The name MUST be a direct, conceptual category tag. Do NOT use poetic, dramatic, or storytelling phrases.
        - GOOD examples: "Nostalgia y madurez", "Amor tóxico", "Crítica a la fama", "Búsqueda de la esencia", "Romance idealizado",
        "Rebeldía juvenil", "Dolor y traición", "Himnos de ruptura", "Enamoramiento idealizado", "Traición y venganza". 
        You can get inspired but these examples, but do not copy them, invent your own based on the keywords.
        - BAD examples (DO NOT DO THIS): "En el umbral de la desesperanza", "Memorias de un amor en fuga", "Amor y libertad perdida",
        "Romance con sombras oscuras", "Recuerdos de la infancia".
        2. LENGTH: MAXIMUM 3-5 WORDS.
        3. OUTPUT FORMAT: ONLY output the exact category name in Spanish. No intros, no quotes, no explanations. 
        """

        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role":"system", "content": "You are a precise music data taxonomy assistant."},
                    {"role":"user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.3,
                max_tokens=15
            )

            llm_name = chat_completion.choices[0].message.content.strip().replace('"', '')
            if ":" in llm_name:
                llm_name = llm_name.split(":")[-1].strip()
            llm_name = llm_name.split('\n')[0].strip()

            names[topic_id] = llm_name
            already_used_names.append(llm_name)
            time.sleep(2)
        
        except Exception as e:
            print(f"Error nombrando clúster {topic_id}: {e}")
            names[topic_id] = f"Cluster {topic_id}"

    df['topic_name_llm'] = df['topic_bertopic'].map(names).fillna("Ruido / Sin clasificar")
    df['topic_words'] = df['topic_bertopic'].map(words_dict).fillna("No aplicable")

    filename = artist_name.lower().replace(" ", "_")
    final_path = f"data/interim/{filename}_definitivo.csv"
    df.to_csv(final_path, index=False)

    print(f"\nPaso 4 completado: ({final_path})")
    return df

def step_5_llm_summarization(df, artist_name):
    print(f"\nPaso 5: Generando resúmenes narrativos con Llama 3 para {artist_name}...")

    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=GROQ_API_KEY)

    if 'llm_meaning' not in df.columns:
        df['llm_meaning'] = None
    if 'llm_validation' not in df.columns:
        df['llm_validation'] = None
    if 'album_confidence' not in df.columns:
        df['album_confidence'] = "0"

    def call_llm_with_retry(prompt, role_context, temperature, max_tokens, max_retries=4):
        wait = 15
        for _ in range(max_retries):
            try:
                answer = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": role_context},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return answer.choices[0].message.content.strip()
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    print(f"Rate limit alcanzado. Esperando {wait} segundos antes de reintentar...")
                    time.sleep(wait)
                    wait *= 2
                else:
                    return f"Error: {str(e)}"
        return None
    
    print(f"Redactando el significado y validando las métricas de {len(df)} canciones...")
    
    failed_songs = []

    for idx, row in df.iterrows():
        title = row['title']
        album = row['album']
        texto_letra = str(row.get('lyrics_full', ''))
        lyrics = ' '.join(texto_letra.split()[:250])
        theme = row.get('theme_zeroshot', 'Desconocido')
        emotion = row.get('emotion_goemotions', 'Desconocida')

        prompt_meaning = f"""
        Analyze the following lyrics from the song '{title}' performed by {artist_name}:
        "{lyrics}"

        Focus exclusively on the narrative, emotions and literary meaning of the text provided.
        Write a single, brief paragraph (maximum 5 lines) IN SPANISH explaining the story behind this song and its main message.
        Respond strictly IN SPANISH.
        """

        prompt_validation = f"""
        Evaluate the accuracy of a Machine Learning classification for the song '{title}' by {artist_name}.
        - Lyrics: "{lyrics}"
        - Predicted emotion by the ML model: {emotion}
        - Predicted theme by the ML model: {theme}

        Write a single, brief paragraph (maximum 5 lines) IN SPANISH explaining if this emotion is accurate for the song's narrative.
        If it is not accurate, suggest a better emotion and briefly explain why, basing your explanation on the lyrics of the song.
        Respond strictly IN SPANISH.
        """

        prompt_album = f"""
        Given the song '{title}' performed by {artist_name}, identify its ORIGINAL studio album.
        As a reference/hint, the Genius API currently lists this song under the project: '{album}', but it may or may not be correct.

        Rules:
        1. ORIGINAL ALBUM: Use your internal knowledge to find the original studio album. RETURN ONLY THE BASE NAME. Exclude any marketing tags, version editions or remaster labels.
        2. UNKNOWN/NEW SONGS: If you do not recognize the song (e.g. it was released after your training data), DO NOT INVENT OR HALLUCINATE an album name. TRUST the Genius API reference '{album}', 
        but clean it semantically by removing any commercial edition/version subtitles, returning only the main album title.
        3. COVERS: If the song is a cover, reply EXACTLY with "Covers".
        4. SINGLES/OTHERS: If the song is a standalone single, soundtrack or you are completely unsure, reply EXACTLY with "Otros".

        You MUST be factual. Do not guess blindly. If you don't know, rely on the provided Genius reference or use "Otros".

        FORMAT REQUIREMENT: You must respond with the album name and your confidence level (0-100) separated by a pipe character "|".
        - Example 1: 1989|100
        - Example 2: The Life Of A Showgirl|60
        - Example 3: Otros|90

        Respond ONLY with the "AlbumName|ConfidenceScore" format. Do not add quotes, spaces around the pipe or explanations.
        """

        ans_meaning = call_llm_with_retry(prompt_meaning, "You are an expert music analyst and literary critic.", temperature=0.7, max_tokens=200)
        if ans_meaning is None:
            print(f"Error por cuota de Groq tras múltiples reintentos. Guardando '{title}' en la cola de reintentos...")
            failed_songs.append(idx)
            continue
        df.at[idx, 'llm_meaning'] = ans_meaning
        time.sleep(2.5)

        ans_validation = call_llm_with_retry(prompt_validation, "You are an expert AI auditor and music emotion analyst.", temperature=0.7, max_tokens=200)
        if ans_validation is None:
            print(f"Error por cuota de Groq tras múltiples reintentos. Guardando '{row['title']}' en la lista para reintentar luego...")
            if idx not in failed_songs:
                failed_songs.append(idx)
            continue
        df.at[idx, 'llm_validation'] = ans_validation
        time.sleep(2.5)

        ans_album = call_llm_with_retry(prompt_album, "You are a strict music cataloging system. You output only exact classification strings.", temperature=0.0, max_tokens=15)
        if ans_album is None:
            print(f"Error por cuota de Groq tras múltiples reintentos. Guardando '{row['title']}' en la lista para reintentar luego...")
            if idx not in failed_songs:
                failed_songs.append(idx)
            continue

        if ans_album and "|" in ans_album:
            parts = ans_album.split("|")
            suggested_album = parts[0].split('\n')[-1].replace('AlbumName:', '').replace('Album:', '').strip()
            confidence = parts[1].split('\n')[0].strip()

            if confidence.isdigit():
                confidence_val = int(confidence)

                if confidence_val < 25:
                    print(f"Baja confianza ({confidence_val}%) en '{title}' (Sugerencia: {suggested_album}). Reasignando a 'Otros'.")
                    df.at[idx, 'album'] = "Otros"
                else:
                    df.at[idx, 'album'] = suggested_album

            else:
                df.at[idx, 'album'] = suggested_album

            df.at[idx, 'album_confidence'] = confidence

        else:
            df.at[idx, 'album'] = "Otros"
            df.at[idx, 'album_confidence'] = "0"
        time.sleep(2.5)

    if len(failed_songs) > 0:
        print(f"\nReprocesando {len(failed_songs)} canciones fallidas...")
        print("Esperando 60s para asegurar que Groq resetea la cuota...")
        time.sleep(60)

        for idx in failed_songs:
            row = df.loc[idx]
            title = row['title']
            album = row['album']
            texto_letra = str(row.get('lyrics_full', ''))
            lyrics = ' '.join(texto_letra.split()[:250])
            theme = row.get('theme_zeroshot', 'Desconocido')
            emotion = row.get('emotion_goemotions', 'Desconocida')

            prompt_meaning = f"""
            Analyze the following lyrics from the song '{title}' performed by {artist_name}:
            "{lyrics}"

            Focus exclusively on the narrative, emotions and literary meaning of the text provided.
            Write a single, brief paragraph (maximum 5 lines) IN SPANISH explaining the story behind this song and its main message.
            Respond strictly IN SPANISH.
            """

            prompt_validation = f"""
            Evaluate the accuracy of a Machine Learning classification for the song '{title}' by {artist_name}.
            - Lyrics: "{lyrics}"
            - Predicted emotion by the ML model: {emotion}
            - Predicted theme by the ML model: {theme}

            Write a single, brief paragraph (maximum 5 lines) IN SPANISH explaining if this emotion is accurate for the song's narrative.
            If it is not accurate, suggest a better emotion and briefly explain why, basing your explanation on the lyrics of the song.
            Respond strictly IN SPANISH.
            """

            prompt_album = f"""
            Given the song '{title}' performed by {artist_name}, identify its ORIGINAL studio album.
            As a reference/hint, the Genius API currently lists this song under the project: '{album}', but it may or may not be correct.

            Rules:
            1. ORIGINAL ALBUM: Use your internal knowledge to find the original studio album. RETURN ONLY THE BASE NAME. Exclude any marketing tags, version editions or remaster labels.
            2. UNKNOWN/NEW SONGS: If you do not recognize the song (e.g. it was released after your training data), DO NOT INVENT OR HALLUCINATE an album name. TRUST the Genius API reference '{album}', 
            but clean it semantically by removing any commercial edition/version subtitles, returning only the main album title.
            3. COVERS: If the song is a cover, reply EXACTLY with "Covers".
            4. SINGLES/OTHERS: If the song is a standalone single, soundtrack or you are completely unsure, reply EXACTLY with "Otros".

            You MUST be factual. Do not guess blindly. If you don't know, rely on the provided Genius reference or use "Otros".

            FORMAT REQUIREMENT: You must respond with the album name and your confidence level (0-100) separated by a pipe character "|".
            - Example 1: 1989|100
            - Example 2: The Life Of A Showgirl|60
            - Example 3: Otros|90

            Respond ONLY with the "AlbumName|ConfidenceScore" format. Do not add quotes, spaces around the pipe or explanations.
            """

            if pd.isna(df.at[idx, 'llm_meaning']):
                df.at[idx, 'llm_meaning'] = call_llm_with_retry(prompt_meaning, "You are an expert music analyst and literary critic.", temperature=0.7, max_tokens=200)
                time.sleep(2.5)

            if pd.isna(df.at[idx, 'llm_validation']):
                df.at[idx, 'llm_validation'] = call_llm_with_retry(prompt_validation, "You are an expert AI auditor and music emotion analyst.", temperature=0.7, max_tokens=200)
                time.sleep(2.5)
            
            ans_album_retry = call_llm_with_retry(prompt_album, "You are a strict music cataloging system. You output only exact classification strings.", temperature=0.0, max_tokens=15)
            if ans_album_retry and "|" in ans_album_retry:
                parts = ans_album_retry.split("|")
                suggested_album = parts[0].split('\n')[-1].replace('AlbumName:', '').replace('Album:', '').strip()
                confidence = parts[1].split('\n')[0].strip()

                if confidence.isdigit():
                    confidence_val = int(confidence)

                    if confidence_val < 25:
                        print(f"Baja confianza ({confidence_val}%) en '{title}' (Sugerencia: {suggested_album}). Reasignando a 'Otros'.")
                        df.at[idx, 'album'] = "Otros"
                    else:
                        df.at[idx, 'album'] = suggested_album
                
                else:
                    df.at[idx, 'album'] = suggested_album
                
                df.at[idx, 'album_confidence'] = confidence

            else:
                df.at[idx, 'album'] = "Otros"
                df.at[idx, 'album_confidence'] = "0"
            time.sleep(2.5)

    filename = artist_name.lower().replace(" ", "_")
    final_path = f"data/processed/{filename}_final.csv"
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(final_path, index=False)

    print(f"Paso 5 completado. CSV final guardado y listo en {final_path}.")
    return df

def enviar_email(destinatario, artista):
    print(f"\nPreparando email de notificación para {destinatario}...")
    try:
        remitente = os.getenv("EMAIL_USER")
        password = os.getenv("EMAIL_PASS")

        if not remitente or not password:
            print("Faltan credenciales EMAIL_USER o EMAIL_PASS en el archivo .env. No se puede enviar el email.")
            return
        
        msg = EmailMessage()
        msg.set_content(f"¡Hola!\n\nEl motor de IA ha terminado de procesar la discografía de {artista.title()}. \n\nYa puedes volver al Dashboard de Streamlit y seleccionarlo en el menú lateral para explorar todas sus métricas, gráficos y análisis literarios.\n\n¡Gracias por usar nuestro sistema!", charset='utf-8')
        msg['Subject'] = f"Análisis de {artista.title()} completado"
        msg['From'] = remitente
        msg['To'] = destinatario

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(remitente, password)
        server.send_message(msg)
        server.quit()
        print("Email enviado con éxito.")
    
    except Exception as e:
        print(f"Error al enviar el email: {e}")

def main():
    parser = argparse.ArgumentParser(description="Pipeline de análisis de discografía musical")
    parser.add_argument("--artist", type=str, required=True, help="Nombre del artista a analizar")
    parser.add_argument("--songs", type=int, default=300, help="Límite máximo de canciones")
    parser.add_argument("--email", type=str, required=False, help="Email para enviar notificación")

    args = parser.parse_args()
    artista = args.artist
    max_songs = args.songs
    email_user = args.email

    print("="*60)
    print(f"Iniciando el procesamiento para: {artista}")
    print("="*60)

    start_time = time.time()

    try:
        df_raw = step_1_data_collection(artista, max_songs=max_songs)
        df_limpio = step_2_preprocessing(df_raw, artista)
        df_metricas = step_3_feature_engineering(df_limpio, artista)
        df_topics = step_4_topic_modeling(df_metricas, artista)
        df_final = step_5_llm_summarization(df_topics, artista)

        print("\nProceso completado con éxito.")

        if email_user:
            enviar_email(email_user, artista)

        end_time = time.time()
        elapsed_time = end_time - start_time

        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        print("="*50)
        if hours > 0:
            print(f"Tiempo total de ejecución: {hours} horas, {minutes} minutos y {seconds} segundos.")
        else:
            print(f"Tiempo total de ejecución: {minutes} minutos y {seconds} segundos.")
        print("="*50)

    except Exception as e:
        print(f"Error durante la ejecución del pipeline: {e}")

if __name__ == "__main__":
    main()
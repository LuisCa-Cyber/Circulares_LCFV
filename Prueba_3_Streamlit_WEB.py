import os
from dotenv import load_dotenv  # Para cargar las variables de entorno
import numpy as np
import pandas as pd
import faiss
import re
import nltk
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unidecode import unidecode
from openai import OpenAI
import streamlit as st

load_dotenv()

# Obtener la clave API desde la variable de entorno
api_key = os.getenv("OPENAI_API_KEY")

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# if api_key is None:
#     st.error("No se encontró la clave API de OpenAI. Verifica las variables de entorno en Streamlit Cloud.")
# else:
#     # Asignar la clave API globalmente
#     openai.api_key = api_key


# # Descargar recursos necesarios de NLTK
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

@st.cache_data
def download_nltk_resources():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

# Llama a la función al inicio
download_nltk_resources()



# Ruta del archivo de embeddings
EMBEDDINGS_FILE = "embeddings.npy"

# Leer el archivo Excel
@st.cache_data
def load_data():
    excel_path = "Chunks.xlsx"  # Cambiar por la ruta de tu archivo
    df = pd.read_excel(excel_path)
    return df

data = load_data()
texts = data["Texto"].tolist()

# Configurar stopwords y lematizador
spanish_stopwords = set(stopwords.words('spanish'))
lemmatizer = nltk.WordNetLemmatizer()

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = unidecode(text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in spanish_stopwords]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

normalized_texts = [normalize_text(text) for text in texts]

############ Parametrizable ############
# Definir get_openai_embedding como función global
def get_openai_embedding(client, text, model="text-embedding-ada-002"):
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# Función para cargar o generar embeddings
@st.cache_data
def load_or_generate_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        # Cargar embeddings desde el archivo si ya existen
        embeddings = np.load(EMBEDDINGS_FILE)
        print("Embeddings cargados desde archivo.")
    else:
        # Generar embeddings si no existen
        embeddings = [get_openai_embedding(client, text) for text in normalized_texts]
        embeddings = np.array(embeddings)
        np.save(EMBEDDINGS_FILE, embeddings)  # Guardar como archivo .npy
        print(f"Embeddings generados y guardados en {EMBEDDINGS_FILE}.")
    return embeddings

# Cargar los embeddings
embeddings = load_or_generate_embeddings()

# Crear índice FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Función de búsqueda
@st.cache_data
def search_similar(query, k=10):
    normalized_query = normalize_text(query)
    query_embedding = np.array([get_openai_embedding(client, normalized_query)])
    distances, indices = index.search(query_embedding, k)
    results = [
        (data.iloc[i]["Texto"], data.iloc[i].get("Circular", "N/A"), data.iloc[i].get("Fecha", "N/A"), distances[0][j])
        for j, i in enumerate(indices[0])
    ]
    return results

# Aplicación Streamlit
@st.cache_data
def get_base_messages():
    return [
         {"role": "system", "content": "Eres un asistente que da apoyo al área comercial. Responde claramente y de forma concisa basándote en los contenidos proporcionados. Siempre indica de qué circular tomaste la información, como ejemplo 'información tomada de CIRCULAR NORMATIVA EXTERNA No. XXX DE XXX', si la respuesta es de varias circulares debes relacionarlas todas."},
         {"role": "system", "content": "Solamente puedes responder sobre los temas asociados a Fondo Nacional de Garantías. Si te preguntan otros temas, debes responder 'No puedo responder tu solicitud, mi conocimiento se basa únicamente en circulares del FNG emitidas en el 2024.'"}        
    ]

#######################################################################
#######################################################################

def run_chatbot():
    # Mostrar el logo y el título en una sección fija
    logo_path = "Imagen2.png"

    # Dividimos la página en columnas para colocar el logo y el título juntos
    with st.container():  # Usar contenedor para que el encabezado se mantenga fijo
        col1, col2 = st.columns([1, 5])  # Ajustar el ancho de las columnas

        # with col1:
        #     st.image(logo_path, width=120)  # Mostrar el logo

        with col1:
            st.markdown(
                """
                 <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
                <img src="""" + logo_path + """" style="width: 120px;">
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                """
                <div style="text-align: center;">
                    <h1 style='color:gray; font-size: 1.8em;'>🤖 Sub. Arquitectura de datos<br>Asistente para CNE 2024 💭</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Inicializar el estado de la sesión si no existe
    if "messages" not in st.session_state:
        st.session_state.messages = get_base_messages()

    # Mostrar todos los mensajes del historial antes de procesar nuevos mensajes
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(f"**Tú:** {content}")
        elif role == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"**ChatBot:** {content}")

    # Campo de entrada para la pregunta del usuario
    query = st.chat_input("¿En qué puedo ayudarte?")

    # Si el usuario escribe una consulta
    if query:
        # Agregar el mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": query})

        # Mostrar el mensaje del usuario inmediatamente
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(f"**Tú:** {query}")

        # Mostrar un indicador animado de "Escribiendo..." por tres ciclos
        typing_placeholder = st.empty()  # Crear un espacio dinámico para el mensaje temporal
        for _ in range(3):  # Repetir el bucle animado tres veces
            for dots in ["", ".", "..", "..."]:  # Animar los puntos
                typing_placeholder.markdown(f"**ChatBot está escribiendo{dots} 🤔**")
                time.sleep(0.2)  # Retardo entre cada cambio

        # Obtener los resultados relevantes (contexto)
        faiss_results = search_similar(query)
        context = "\n".join([
            f"Circular: {circular}, Fecha: {fecha}, Texto: {result_text}"
            for result_text, circular, fecha, _ in faiss_results
        ])

        st.session_state.messages.append(
            {"role": "system", "content": f"Los siguientes documentos son relevantes para la consulta:\n{context}"}
        )

        # Llamada a OpenAI para generar la respuesta del asistente
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # gpt-4o-mini - gpt-3.5-turbo 
            messages=st.session_state.messages,
            temperature=1
        )

        # Obtener la respuesta del asistente
        content = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": content})

        # Reemplazar el indicador de "Escribiendo..." con la respuesta final
        typing_placeholder.empty()  # Eliminar el indicador temporal
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()  # Crear espacio dinámico para el efecto de escritura
            partial_content = ""  # Inicializar contenido parcial
            for char in content:
                partial_content += char
                placeholder.markdown(f"**ChatBot:** {partial_content}")
                time.sleep(0.005)  # Retardo entre caracteres
            placeholder.markdown(f"**ChatBot:** {partial_content}")  # Mostrar el contenido completo al final


if __name__ == "__main__":
    run_chatbot()

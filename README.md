# 🤖 Asistente Virtual FNG - Circulares Normativas 2024

<p align="center">
  <img src="https://portalqa.fng.gov.co/images/institucional/logo_nfg_blanco.png" alt="Logo FNG" style="width: 200px; height: auto;">
</p>

## 📋 Descripción

Este proyecto implementa un asistente virtual basado en IA para consultar información sobre las Circulares Normativas Externas (CNE) 2024 del Fondo Nacional de Garantías (FNG). Utiliza tecnologías avanzadas de procesamiento de lenguaje natural y búsqueda semántica para proporcionar respuestas precisas y contextualizadas.

## ⚡ Características Principales

- 🔍 Búsqueda semántica utilizando FAISS
- 💬 Interfaz conversacional intuitiva con Streamlit
- 🧠 Procesamiento de lenguaje natural con NLTK
- 🔄 Sistema de caché para optimizar el rendimiento
- 📊 Integración con OpenAI para generación de respuestas
- 🎯 Normalización y procesamiento de texto avanzado

## 🛠️ Tecnologías Utilizadas

- Python 3.x
- Streamlit
- OpenAI API
- FAISS (Facebook AI Similarity Search)
- NLTK (Natural Language Toolkit)
- Pandas
- NumPy

## 📦 Requisitos de Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/LuisCa-Cyber/Circulares_LCFV.git
```

2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

3. Configurar las variables de entorno:
   - Crear un archivo `.env` en la raíz del proyecto
   - Agregar la clave API de OpenAI:
```bash
OPENAI_API_KEY=tu_clave_api
```

## 🚀 Uso

1. Ejecutar la aplicación:
```bash
streamlit run Prueba_3_Streamlit_WEB.py
```

2. Acceder a la interfaz web a través del navegador (por defecto en `localhost:8501`)

## 🏗️ Estructura del Proyecto

```
├── Prueba_3_Streamlit_WEB.py   # Archivo principal de la aplicación
├── Chunks.xlsx                 # Base de datos de circulares
├── embeddings.npy             # Archivo de embeddings pre-calculados
├── Imagen2.png                # Logo de la aplicación
├── requirements.txt           # Dependencias del proyecto
└── .env                       # Variables de entorno (no incluido en el repo)
```

## 🔧 Funcionalidades Principales

- **Procesamiento de Texto**: Normalización, eliminación de stopwords y lematización
- **Generación de Embeddings**: Utiliza el modelo `text-embedding-ada-002` de OpenAI
- **Búsqueda Semántica**: Implementada con FAISS para recuperación eficiente
- **Interfaz de Usuario**: Diseño responsivo y amigable con Streamlit
- **Sistema de Caché**: Optimización de recursos y tiempo de respuesta

## 👥 Contribución

Las contribuciones son bienvenidas. Por favor, sigue estos pasos:

1. Fork el proyecto
2. Crea una nueva rama (`git checkout -b feature/NuevaFuncionalidad`)
3. Realiza tus cambios
4. Commit tus cambios (`git commit -m 'Agregar nueva funcionalidad'`)
5. Push a la rama (`git push origin feature/NuevaFuncionalidad`)
6. Abre un Pull Request

## 📞 Contacto

Luis Carlos Fernández - Ingeniero de Datos e IA en Fondo Nacional de Garantías

Correo empresa: Luis.Fernandez@fng.gov.co

Correo personal: luiscafer728@hotmail.com - luisfernandezv728@gmail.com

Link del proyecto: [https://github.com/LuisCa-Cyber/Circulares_LCFV](https://github.com/LuisCa-Cyber/Circulares_LCFV)

---
<p align="center">
  Desarrollado con ❤️ por la Subdirección de Arquitectura de Datos - FNG
</p>

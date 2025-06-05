# 🤖 Chatbot AI Multimodelo con Personalidades y Roles

Este es un chatbot inteligente en línea de comandos (CLI), diseñado para adaptarse al contexto del usuario mediante la selección de:

- 🔍 **Modelos LLM** (OpenAI y Ollama)
- 🧠 **Roles profesionales** (ej. Científico, Médico, Abogado)
- 🎭 **Personalidades** (ej. Crítico, Amable, Zen)

Ofrece una experiencia conversacional versátil, personalizable y lista para ser ejecutada localmente o en entornos Docker.

---

## 🚀 Características

✅ Modelos disponibles:

- ChatGPT (`gpt-4o-mini`)
- DeepSeek-R1 (`deepseek-r1:14b`)
- Gemma3 (`gemma3:27b`)
- Llama3 (`llama3:8b`)

✅ Combinación dinámica:

- `Rol profesional` + `Estilo de personalidad`

✅ Modo conversación con memoria (contexto acumulativo)

✅ Comandos interactivos:
- `cambiar modelo`
- `cambiar personalidad`
- `cambiar rol`
- `salir`

---

## 🖥️ Requisitos

- Python 3.10+
- Docker (opcional)
- [Ollama](https://ollama.com/) instalado y corriendo en local (para modelos locales)
- API Key de OpenAI (si usas modelos de OpenAI)

---

## 📁 Estructura del proyecto

```
chatbot-ai/
├── app.py                   # Script principal
├── prompts/
│   ├── modelos.json         # Modelos disponibles y sus tipos
│   ├── personalidades.json  # Estilos de comunicación
│   └── roles.json           # Conocimientos y perfiles profesionales
├── Dockerfile               # Contenedor reproducible
├── .dockerignore            # Ignorados por Docker
├── .gitignore               # Ignorados por Git
├── requirements.txt         # Dependencias del proyecto
└── README.md
```

---

## ⚙️ Instalación local

```bash
# Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate  # o venv\Scripts\activate en Windows

# Instalar dependencias
pip install -r requirements.txt

# Crear archivo .env con tu API Key de OpenAI
echo "OPENAI_API_KEY=sk-xxx..." > .env

# Ejecutar chatbot
python app.py
```

---

## 🐳 Uso con Docker

### 🔧 Construir imagen

```bash
docker build -t chatbot-ai .
```

### ▶️ Ejecutar contenedor

```bash
docker run -it chatbot-ai
```

### 🧪 Montar `.env` y prompts personalizados

```bash
docker run -it \
  -v $(pwd)/.env:/app/.env \
  -v $(pwd)/prompts:/app/prompts \
  chatbot-ai
```

> En Windows reemplaza `$(pwd)` por la ruta completa, ej. `C:\Users\tu_usuario\proyecto\chatbot-ai`

---

## 🔐 Configuración `.env`

Crea un archivo `.env` en la raíz del proyecto con tu clave de API:

```
OPENAI_API_KEY=sk-xxxxxxx
```

---

## 🧠 Roles y Personalidades

Los archivos JSON definen comportamientos y especialidades. Puedes editar o expandir:

- `prompts/personalidades.json`
- `prompts/roles.json`

Cada entrada define estilo, saludo, ramas y nivel técnico.

---

## 🛠️ Dependencias principales

- `openai` (para modelos GPT)
- `requests` (para comunicar con Ollama)
- `python-dotenv` (para leer variables del entorno)

Instalables vía:

```bash
pip install -r requirements.txt
```

---

## 📜 Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT.

---
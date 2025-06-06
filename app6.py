# app6.py – Generador de Imágenes Multimodelo
# -------------------------------------------
# Este script permite generar imágenes usando múltiples modelos:
# - DALL·E 3 de OpenAI vía API
# - Stable Diffusion XL y OpenJourney v4 (locales, vía HuggingFace Diffusers)
# Las imágenes generadas se guardan en una carpeta local "images".

# ==============================
# 📦 IMPORTACIONES Y CONFIGURACIÓN
# ==============================

import os
import uuid
import datetime
import requests
from dotenv import load_dotenv

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import openai

# Cargar variables de entorno desde .env (ej. API key de OpenAI)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Crear carpeta de salida de imágenes si no existe
os.makedirs("images", exist_ok=True)

# ==============================
# ⚙️ FUNCIONES DE CARGA DE MODELOS
# ==============================

def cargar_sdxl():
    """
    Carga el modelo local Stable Diffusion XL desde el directorio 'models/sdxl'.
    Requiere GPU con soporte FP16.
    """
    print("🚀 Cargando modelo SDXL...")
    return StableDiffusionXLPipeline.from_pretrained(
        "models/sdxl",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

def cargar_openjourney():
    """
    Carga el modelo OpenJourney v4 desde el directorio 'models/openjourney'.
    Este modelo tiene un estilo visual similar a MidJourney.
    """
    print("🎨 Cargando modelo OpenJourney...")
    return StableDiffusionPipeline.from_pretrained(
        "models/openjourney",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

# ==============================
# 🧠 GENERACIÓN DE IMÁGENES
# ==============================

def generar_con_openai(prompt: str) -> str:
    """
    Solicita la generación de una imagen a la API de OpenAI (DALL·E 3).
    
    Parámetros:
    - prompt: descripción de la imagen.
    
    Retorna:
    - URL de la imagen generada.
    """
    print("📡 Solicitando imagen a DALL·E 3 (OpenAI)...")
    response = openai.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="hd",
        n=1
    )
    return response.data[0].url

def descargar_y_guardar_imagen(url: str, modelo: str, prompt: str) -> str:
    """
    Descarga la imagen generada por DALL·E 3 a partir de la URL proporcionada.
    La guarda localmente en la carpeta 'images/'.
    """
    nombre = f"{modelo}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
    ruta = os.path.join("images", nombre)
    try:
        response = requests.get(url)
        with open(ruta, "wb") as f:
            f.write(response.content)
        print(f"✅ Imagen descargada y guardada: {ruta}")
        return ruta
    except Exception as e:
        print(f"❌ Error al guardar imagen: {e}")
        return ""

def generar_con_diffusers(pipe, prompt: str) -> Image.Image:
    """
    Genera una imagen con un modelo Diffusers (Stable Diffusion o OpenJourney).
    
    Parámetros:
    - pipe: pipeline del modelo cargado previamente.
    - prompt: descripción de la imagen.
    
    Retorna:
    - imagen PIL generada.
    """
    result = pipe(prompt)
    return result.images[0]

def guardar_imagen(imagen: Image.Image, modelo: str, prompt: str) -> str:
    """
    Guarda una imagen PIL en la carpeta 'images/' con nombre único.
    
    Parámetros:
    - imagen: objeto PIL.
    - modelo: identificador del modelo usado.
    - prompt: texto descriptivo original (solo para naming).
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{modelo}_{timestamp}_{uuid.uuid4().hex[:6]}.png"
    ruta = os.path.join("images", filename)
    imagen.save(ruta)
    print(f"✅ Imagen guardada: {ruta}")
    return ruta

# ==============================
# 📋 MENÚ DE SELECCIÓN DE MODELO
# ==============================

def seleccionar_modelo():
    """
    Muestra el menú de selección de modelo para el usuario.
    
    Retorna:
    - Opción seleccionada como string.
    """
    print("\nSelecciona un modelo:")
    print("1. 🎨 DALL·E 3 (OpenAI)")
    print("2. 🧠 Stable Diffusion XL")
    print("3. 🌀 OpenJourney v4")
    print("4. ❌ Salir")
    return input("Ingresa el número del modelo: ").strip()

# ==============================
# 🧵 FLUJO PRINCIPAL DEL SCRIPT
# ==============================

if __name__ == "__main__":
    print("🧠 Generador de Imágenes Multimodelo")
    print("Escribe 'salir' en cualquier momento para terminar.\n")

    modelo_actual = None  # Identificador textual del modelo actual
    pipe = None           # Pipeline cargado (solo para modelos locales)

    while True:
        eleccion = seleccionar_modelo()

        if eleccion == "1":
            modelo_actual = "dalle3"
            pipe = None
        elif eleccion == "2":
            modelo_actual = "sdxl"
            pipe = cargar_sdxl()
        elif eleccion == "3":
            modelo_actual = "openjourney"
            pipe = cargar_openjourney()
        elif eleccion == "4":
            print("👋 Hasta luego.")
            break
        else:
            print("❌ Opción inválida. Intenta de nuevo.")
            continue

        while True:
            prompt = input("\n📝 Describe la imagen que deseas generar:\n> ").strip()
            if prompt.lower() in ["salir", "exit"]:
                print("👋 Saliendo del modo de generación.\n")
                break

            print("🧠 Generando imagen...")

            try:
                if modelo_actual == "dalle3":
                    url = generar_con_openai(prompt)
                    print(f"🌐 Imagen generada:\n{url}\n")
                    descargar_y_guardar_imagen(url, modelo_actual, prompt)
                else:
                    imagen = generar_con_diffusers(pipe, prompt)
                    guardar_imagen(imagen, modelo_actual, prompt)
            except Exception as e:
                print(f"❌ Error generando imagen: {e}")

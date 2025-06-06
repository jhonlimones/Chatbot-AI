# app6.py – Generador de Imágenes Multimodelo

import os
import uuid
import datetime
import requests
from dotenv import load_dotenv

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import openai

# ========== CONFIGURACIÓN ==========
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

os.makedirs("images", exist_ok=True)

# ========== FUNCIONES DE CARGA ==========
def cargar_sdxl():
    print("🚀 Cargando modelo SDXL...")
    return StableDiffusionXLPipeline.from_pretrained(
        "models/sdxl",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

def cargar_openjourney():
    print("🎨 Cargando modelo OpenJourney...")
    return StableDiffusionPipeline.from_pretrained(
        "models/openjourney",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

# ========== GENERADORES ==========
def generar_con_openai(prompt: str) -> str:
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
    result = pipe(prompt)
    return result.images[0]

def guardar_imagen(imagen: Image.Image, modelo: str, prompt: str) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{modelo}_{timestamp}_{uuid.uuid4().hex[:6]}.png"
    ruta = os.path.join("images", filename)
    imagen.save(ruta)
    print(f"✅ Imagen guardada: {ruta}")
    return ruta

# ========== MENÚ ==========
def seleccionar_modelo():
    print("\nSelecciona un modelo:")
    print("1. 🎨 DALL·E 3 (OpenAI)")
    print("2. 🧠 Stable Diffusion XL")
    print("3. 🌀 OpenJourney v4")
    print("4. ❌ Salir")
    return input("Ingresa el número del modelo: ").strip()

# ========== MAIN ==========
if __name__ == "__main__":
    print("🧠 Generador de Imágenes Multimodelo")
    print("Escribe 'salir' en cualquier momento para terminar.\n")

    modelo_actual = None
    pipe = None

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

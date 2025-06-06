import openai
import os
import requests
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

# Cargar API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Ruta de la carpeta donde se guardarán las imágenes
IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(exist_ok=True)

# Generar imagen con DALL·E 3 y devolver URL
def generar_imagen(prompt: str) -> str:
    response = openai.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="hd",
        n=1
    )
    return response.data[0].url

# Descargar y guardar la imagen en disco
def guardar_imagen(url: str, modelo: str = "dalle3") -> str:
    try:
        respuesta = requests.get(url)
        respuesta.raise_for_status()
        nombre_archivo = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{modelo}.png"
        ruta_completa = IMAGES_DIR / nombre_archivo
        with open(ruta_completa, "wb") as f:
            f.write(respuesta.content)
        return str(ruta_completa)
    except Exception as e:
        raise RuntimeError(f"No se pudo guardar la imagen: {e}")

# Programa principal
if __name__ == "__main__":
    print("🎨 Generador de Imágenes con DALL·E (OpenAI)")
    print("Escribe tu descripción o 'salir' para terminar.\n")

    while True:
        prompt = input("Describe la imagen que deseas generar:\n> ").strip()
        if prompt.lower() in ["salir", "exit"]:
            print("Hasta luego.")
            break

        print("🧠 Generando imagen, espera un momento...\n")
        try:
            url = generar_imagen(prompt)
            ruta = guardar_imagen(url)
            print(f"✅ Imagen generada y guardada en:\n{ruta}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")

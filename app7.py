# app7.py – Generador de Imágenes Multimodelo (con mejora de prompts)
# ----------------------------------------------------------
# Este script permite generar imágenes con múltiples modelos:
# - DALL·E 3 de OpenAI vía API
# - Stable Diffusion XL y OpenJourney v4 (locales, vía HuggingFace Diffusers)
# - Mejora de prompt con modelo local DeepSeek-R1 (Ollama)
# Las imágenes se guardan en "images/" y los prompts mejorados en "prompts/generados"

# ==============================
# 📦 IMPORTACIONES Y CONFIGURACIÓN
# ==============================

import os
import uuid
import datetime
import requests
import json
from dotenv import load_dotenv

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import openai

# Cargar variables de entorno
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# Crear carpetas necesarias
os.makedirs("images", exist_ok=True)
os.makedirs("prompts/generados", exist_ok=True)

# Ruta al prompt base de mejora
PROMPT_PATH = "prompts/prompt_mejorador.json"

# ==============================
# ⚙️ FUNCIONES UTILITARIAS
# ==============================

def verificar_ollama():
    """Verifica si Ollama está corriendo"""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def verificar_modelo_disponible():
    """Verifica si el modelo DeepSeek-R1 está disponible"""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            modelos = response.json().get("models", [])
            return any(OLLAMA_MODEL in modelo.get("name", "") for modelo in modelos)
        return False
    except:
        return False

def cargar_prompt_base():
    """Carga el prompt base para mejorar prompts"""
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["instruccion"]

def mejorar_prompt_con_deepseek(user_prompt: str) -> str:
    """Mejora un prompt usando DeepSeek-R1"""
    base = cargar_prompt_base()
    completo = f"{base}\n\nUsuario: {user_prompt}"

    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": OLLAMA_MODEL, 
            "prompt": completo, 
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 300
            }
        },
        timeout=60
    )
    response.raise_for_status()
    resultado = response.json().get("response", "").strip()

    if not resultado:
        raise ValueError("La respuesta del modelo está vacía.")

    # Limpiar respuesta de etiquetas de pensamiento
    if "<think>" in resultado:
        if "</think>" in resultado:
            resultado = resultado.split("</think>")[-1].strip()
        else:
            resultado = resultado.split("<think>")[0].strip()
    
    # Limpiar posibles prefijos no deseados
    prefijos_a_remover = ["Prompt:", "prompt:", "Nuevo prompt:", "Mejorado:", "Descripción:"]
    for prefijo in prefijos_a_remover:
        if resultado.startswith(prefijo):
            resultado = resultado[len(prefijo):].strip()
    
    return resultado

def guardar_prompt_mejorado(prompt_original: str, prompt_mejorado: str):
    """Guarda el prompt mejorado en un archivo JSON"""
    nombre = f"prompt_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    ruta = os.path.join("prompts", "generados", nombre)
    data = {
        "fecha": datetime.datetime.now().isoformat(),
        "prompt_original": prompt_original,
        "prompt_mejorado": prompt_mejorado
    }
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"📝 Prompt guardado: {ruta}")

# ==============================
# 🎨 FUNCIONES DE MODELOS
# ==============================

def cargar_sdxl():
    """Carga el modelo Stable Diffusion XL"""
    print("🚀 Cargando modelo SDXL...")
    return StableDiffusionXLPipeline.from_pretrained(
        "models/sdxl",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

def cargar_openjourney():
    """Carga el modelo OpenJourney"""
    print("🎨 Cargando modelo OpenJourney...")
    return StableDiffusionPipeline.from_pretrained(
        "models/openjourney",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

def generar_con_openai(prompt: str, es_mejorado: bool = False) -> str:
    """Genera imagen con DALL·E 3"""
    print("📡 Solicitando imagen a DALL·E 3 (OpenAI)...")
    response = openai.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="hd",
        n=1
    )
    url = response.data[0].url
    prefijo = "mejorado_dalle3" if es_mejorado else "dalle3"
    return descargar_y_guardar_imagen(url, prefijo)

def descargar_y_guardar_imagen(url: str, modelo: str) -> str:
    """Descarga y guarda una imagen desde URL"""
    nombre = f"{modelo}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
    ruta = os.path.join("images", nombre)
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(ruta, "wb") as f:
            f.write(response.content)
        print(f"✅ Imagen descargada: {ruta}")
        return ruta
    except Exception as e:
        print(f"❌ Error al guardar imagen: {e}")
        return ""

def generar_con_diffusers(pipe, prompt: str, modelo_name: str, es_mejorado: bool = False) -> str:
    """Genera imagen con modelos Diffusers (SDXL o OpenJourney)"""
    print(f"🎨 Generando imagen con {modelo_name}...")
    imagen = pipe(prompt).images[0]
    prefijo = f"mejorado_{modelo_name}" if es_mejorado else modelo_name
    nombre = f"{prefijo}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
    ruta = os.path.join("images", nombre)
    imagen.save(ruta)
    print(f"✅ Imagen generada y guardada: {ruta}")
    return ruta

# ==============================
# 📋 MENÚS
# ==============================

def menu_principal():
    """Muestra el menú principal"""
    print("\n" + "="*50)
    print("🧠 GENERADOR DE IMÁGENES MULTIMODELO")
    print("="*50)
    print("1. 🎨 DALL·E 3 (OpenAI)")
    print("2. 🧠 Stable Diffusion XL")
    print("3. 🎭 OpenJourney v4")
    print("4. ✨ Mejorar un prompt")
    print("5. ❌ Salir")
    print("="*50)
    return input("Elige una opción: ").strip()

def menu_decision_prompt():
    """Menú para decidir qué hacer con el prompt mejorado"""
    print("\n" + "-"*40)
    print("¿Qué deseas hacer?")
    print("-"*40)
    print("A) Usar este prompt mejorado")
    print("B) Introducir otro prompt")
    print("C) Volver al menú principal")
    print("-"*40)
    return input("Elige una opción (A/B/C): ").strip().upper()

def menu_modelos_para_generar():
    """Menú para seleccionar modelo de generación"""
    print("\n" + "-"*40)
    print("Selecciona el modelo para generar:")
    print("-"*40)
    print("1. 🎨 DALL·E 3 (OpenAI)")
    print("2. 🧠 Stable Diffusion XL")
    print("3. 🎭 OpenJourney v4")
    print("-"*40)
    return input("Elige una opción: ").strip()

# ==============================
# 🔄 FLUJOS DE TRABAJO
# ==============================

def flujo_generacion_directa(opcion: str):
    """Flujo para generación directa sin mejora de prompt"""
    modelos = {
        "1": ("DALL·E 3", "dalle3"),
        "2": ("Stable Diffusion XL", "sdxl"),
        "3": ("OpenJourney v4", "openjourney")
    }
    
    nombre_modelo, codigo_modelo = modelos[opcion]
    prompt = input(f"\n📝 Describe la imagen para {nombre_modelo}:\n> ").strip()
    
    if not prompt:
        print("❌ No se proporcionó ningún prompt.")
        return
    
    try:
        if opcion == "1":
            generar_con_openai(prompt)
        elif opcion == "2":
            pipe = cargar_sdxl()
            generar_con_diffusers(pipe, prompt, "sdxl")
        elif opcion == "3":
            pipe = cargar_openjourney()
            generar_con_diffusers(pipe, prompt, "openjourney")
    except Exception as e:
        print(f"❌ Error al generar imagen: {e}")

def flujo_mejora_prompt():
    """Flujo completo para mejorar prompts"""
    if not verificar_ollama():
        print("❌ Ollama no está disponible. No se puede mejorar prompts.")
        return
    
    if not verificar_modelo_disponible():
        print(f"❌ El modelo {OLLAMA_MODEL} no está disponible.")
        return
    
    while True:  # Loop para permitir introducir múltiples prompts
        print("\n" + "="*50)
        print("✨ MEJORADOR DE PROMPTS")
        print("="*50)
        
        prompt_original = input("📝 Escribe tu prompt para mejorar:\n> ").strip()
        
        if not prompt_original:
            print("❌ No se proporcionó ningún prompt.")
            continue
        
        print("🤔 Mejorando prompt con DeepSeek-R1...")
        
        try:
            prompt_mejorado = mejorar_prompt_con_deepseek(prompt_original)
            
            print(f"\n{'='*50}")
            print("📝 RESULTADO:")
            print("="*50)
            print(f"🔸 Original: {prompt_original}")
            print(f"✨ Mejorado: {prompt_mejorado}")
            print("="*50)
            
            # Guardar automáticamente
            guardar_prompt_mejorado(prompt_original, prompt_mejorado)
            
            # Menú de decisión
            decision = menu_decision_prompt()
            
            if decision == "A":
                # Usar prompt mejorado
                modelo_opcion = menu_modelos_para_generar()
                
                try:
                    if modelo_opcion == "1":
                        generar_con_openai(prompt_mejorado, es_mejorado=True)
                    elif modelo_opcion == "2":
                        pipe = cargar_sdxl()
                        generar_con_diffusers(pipe, prompt_mejorado, "sdxl", es_mejorado=True)
                    elif modelo_opcion == "3":
                        pipe = cargar_openjourney()
                        generar_con_diffusers(pipe, prompt_mejorado, "openjourney", es_mejorado=True)
                    else:
                        print("❌ Opción inválida.")
                        continue
                    
                    print("✅ Imagen generada exitosamente!")
                    break  # Salir del loop y volver al menú principal
                    
                except Exception as e:
                    print(f"❌ Error al generar imagen: {e}")
                    break
            
            elif decision == "B":
                # Introducir otro prompt - continúa el loop
                continue
                
            elif decision == "C":
                # Volver al menú principal
                break
            else:
                print("❌ Opción inválida. Volviendo al menú principal...")
                break
                
        except Exception as e:
            print(f"❌ Error al mejorar prompt: {e}")
            break

# ==============================
# 🚀 PROGRAMA PRINCIPAL
# ==============================

def inicializar_sistema():
    """Inicializa y verifica el sistema"""
    print("🧠 Generador de Imágenes Multimodelo + Mejorador de Prompts")
    print("Inicializando sistema...")
    
    # Verificar configuración
    if not openai.api_key:
        print("⚠️  OPENAI_API_KEY no configurada - DALL·E 3 no estará disponible")
    else:
        print("✅ OpenAI API configurada")
    
    if not verificar_ollama():
        print("⚠️  Ollama no está corriendo - Mejora de prompts no disponible")
    elif not verificar_modelo_disponible():
        print(f"⚠️  El modelo {OLLAMA_MODEL} no está disponible")
    else:
        print(f"✅ DeepSeek-R1 listo para mejorar prompts")
    
    print("✅ Sistema iniciado\n")

if __name__ == "__main__":
    inicializar_sistema()
    
    while True:
        try:
            opcion = menu_principal()
            
            if opcion in ["1", "2", "3"]:
                flujo_generacion_directa(opcion)
                
            elif opcion == "4":
                flujo_mejora_prompt()
                
            elif opcion == "5":
                print("\n👋 ¡Hasta luego!")
                break
                
            else:
                print("❌ Opción inválida. Por favor, elige una opción del 1 al 5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Programa interrumpido por el usuario. ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            print("Continuando...")
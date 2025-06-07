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
    """Carga el prompt base para mejorar prompts desde el archivo JSON"""
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["instruccion"]

def validar_prompt_completo(prompt: str) -> bool:
    """Valida si un prompt está completo y no cortado"""
    if not prompt or len(prompt.strip()) < 10:
        return False
    
    # Verificar si termina abruptamente (sin puntuación adecuada)
    prompt_limpio = prompt.strip()
    
    # Si termina en medio de una palabra o frase
    palabras_incompletas = [
        "en el", "en la", "con un", "con una", "de la", "de un", 
        "que está", "mientras", "durante", "bajo un", "sobre la", 
        "hacia", "desde"
    ]
    
    for palabra_incompleta in palabras_incompletas:
        if prompt_limpio.lower().endswith(palabra_incompleta):
            return False
    
    # Si no termina con puntuación o palabra completa
    if prompt_limpio.endswith((",", "en", "y", "con", "de", "la", "el", "un", "una")):
        return False
        
    return True

def mejorar_prompt_con_deepseek(user_prompt: str) -> str:
    """Mejora un prompt usando DeepSeek-R1 con reintentos y validación mejorada"""
    max_intentos = 3
    
    for intento in range(max_intentos):
        try:
            print(f"🔄 Intento {intento + 1}/{max_intentos}")
            
            # Cargar prompt base desde el archivo JSON
            base = cargar_prompt_base()
            
            # Crear un prompt más específico para evitar cortes
            prompt_mejorado = f"""{base}

IMPORTANTE: Debes completar TODA la descripción. No te detengas a mitad de frase. Asegúrate de terminar con una descripción completa y coherente.

Usuario: {user_prompt}

Descripción mejorada:"""

            # Parámetros más agresivos para obtener respuestas completas
            payload = {
                "model": OLLAMA_MODEL, 
                "prompt": prompt_mejorado, 
                "stream": False,
                "options": {
                    "temperature": 0.9,
                    "top_p": 0.95,
                    "top_k": 40,
                    "num_predict": 800,        # Aumentado significativamente
                    "repeat_penalty": 1.1,
                    "stop": [],                # Sin stops para evitar cortes prematuros
                    "num_ctx": 4096           # Contexto más grande
                }
            }
            
            print(f"📡 Enviando solicitud a Ollama (timeout: 120s)...")
            start_time = datetime.datetime.now()
            
            response = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json=payload,
                timeout=120  # Aumentado a 2 minutos
            )
            
            end_time = datetime.datetime.now()
            tiempo_transcurrido = (end_time - start_time).total_seconds()
            print(f"⏱️  Respuesta recibida en {tiempo_transcurrido:.1f} segundos")
            
            response.raise_for_status()
            resultado = response.json().get("response", "").strip()
            
            print(f"📊 Respuesta cruda recibida: {len(resultado)} caracteres")

            if not resultado:
                print(f"⚠️  Intento {intento + 1}: Respuesta vacía del modelo")
                continue

            # Limpiar respuesta de etiquetas de pensamiento (más exhaustivo)
            resultado_original = resultado
            
            # Procesar etiquetas <think>
            if "<think>" in resultado:
                print("🧹 Limpiando etiquetas de pensamiento...")
                
                if "</think>" in resultado:
                    # Tomar todo lo que esté después de </think>
                    partes = resultado.split("</think>")
                    if len(partes) > 1:
                        resultado = partes[-1].strip()
                    else:
                        resultado = resultado.split("<think>")[0].strip()
                else:
                    # Si no hay cierre, tomar lo que esté antes de <think>
                    resultado = resultado.split("<think>")[0].strip()
            
            # Limpiar prefijos no deseados (más exhaustivo)
            prefijos_a_remover = [
                "Prompt:", "prompt:", "Nuevo prompt:", "Mejorado:", "Descripción:", 
                "Descripción mejorada:", "descripción mejorada:", "Resultado:",
                "resultado:", "Aquí tienes:", "aquí tienes:", "Te propongo:",
                "te propongo:", "Sería:", "sería:"
            ]
            
            for prefijo in prefijos_a_remover:
                if resultado.startswith(prefijo):
                    resultado = resultado[len(prefijo):].strip()
            
            # Remover comillas si envuelven toda la respuesta
            if resultado.startswith('"') and resultado.endswith('"'):
                resultado = resultado[1:-1].strip()
            
            print(f"📊 Después de limpieza: {len(resultado)} caracteres")
            
            # Validación mejorada de completitud
            if not validar_prompt_completo_mejorado(resultado):
                print(f"⚠️  Intento {intento + 1}: Prompt parece incompleto")
                print(f"🔍 Terminación actual: '...{resultado[-50:]}'")
                continue
            
            print(f"✅ Prompt mejorado exitosamente en intento {intento + 1}")
            return resultado
            
        except requests.exceptions.Timeout:
            print(f"⏰ Intento {intento + 1}: Timeout - el modelo tardó más de 120 segundos")
            continue
        except Exception as e:
            print(f"⚠️  Intento {intento + 1} falló: {e}")
            if intento == max_intentos - 1:
                raise ValueError(f"No se pudo mejorar el prompt después de {max_intentos} intentos")
            continue
    
    raise ValueError("Se agotaron todos los intentos para mejorar el prompt")

def validar_prompt_completo_mejorado(prompt: str) -> bool:
    """Validación más estricta para detectar prompts incompletos"""
    if not prompt or len(prompt.strip()) < 20:  # Mínimo más alto
        return False
    
    prompt_limpio = prompt.strip()
    
    # Verificar longitud mínima esperada para un prompt mejorado
    if len(prompt_limpio) < 50:
        return False
    
    # Palabras/frases que indican final abrupto
    finales_abruptos = [
        # Preposiciones
        "en el", "en la", "con un", "con una", "de la", "de un", "por el", "por la",
        "hacia el", "hacia la", "desde el", "desde la", "sobre el", "sobre la",
        "bajo el", "bajo la", "entre el", "entre la", "durante el", "durante la",
        
        # Conjunciones y conectores
        "que está", "que se", "mientras", "durante", "cuando", "donde", "como",
        "y", "pero", "sin embargo", "además", "también",
        
        # Artículos y determinantes
        "el", "la", "los", "las", "un", "una", "unos", "unas",
        
        # Verbos auxiliares incompletos
        "está", "son", "tienen", "puede", "debe", "va", "viene",
        
        # Palabras que sugieren continuación
        "creando", "generando", "mostrando", "revelando", "capturando",
        "destacando", "iluminando", "reflejando"
    ]
    
    for final_abrupto in finales_abruptos:
        if prompt_limpio.lower().endswith(final_abrupto.lower()):
            return False
    
    # Verificar si termina con coma (generalmente indica continuación)
    if prompt_limpio.endswith(","):
        return False
    
    # Verificar si la última palabra está incompleta (muy corta sin puntuación)
    ultima_palabra = prompt_limpio.split()[-1] if prompt_limpio.split() else ""
    if len(ultima_palabra) < 3 and not ultima_palabra.endswith(('.', '!', '?')):
        return False
    
    # Verificar que tenga al menos algunos elementos descriptivos básicos
    elementos_descriptivos = ["color", "luz", "estilo", "atmósfera", "detalle", "textura", "fondo", "primer plano"]
    tiene_elementos = any(elemento in prompt_limpio.lower() for elemento in elementos_descriptivos)
    
    if not tiene_elementos and len(prompt_limpio) < 100:
        return False
    
    return True

def guardar_prompt_mejorado(prompt_original: str, prompt_mejorado: str):
    """Guarda el prompt mejorado en un archivo JSON solo si es válido"""
    # Validar antes de guardar con la nueva función
    if not prompt_mejorado or not validar_prompt_completo_mejorado(prompt_mejorado):
        print("❌ No se guardará el prompt porque está incompleto o vacío")
        return False
        
    nombre = f"prompt_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    ruta = os.path.join("prompts", "generados", nombre)
    data = {
        "fecha": datetime.datetime.now().isoformat(),
        "prompt_original": prompt_original,
        "prompt_mejorado": prompt_mejorado,
        "longitud_caracteres": len(prompt_mejorado),
        "longitud_palabras": len(prompt_mejorado.split()),
        "es_completo": validar_prompt_completo_mejorado(prompt_mejorado)
    }
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"📝 Prompt guardado exitosamente: {ruta}")
    return True

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
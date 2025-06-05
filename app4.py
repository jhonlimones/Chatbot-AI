# Un Chatbot que utiliza OpenAI y Ollama para responder preguntas, 
# con selección de modelos, personalidades y roles profesionales.

import os
import json
import openai
import requests
from dotenv import load_dotenv

# Cargar API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Rutas a archivos de configuración
MODELS_PATH = "prompts/models.json"
PERSONALIDADES_PATH = "prompts/personalidades.json"
ROLES_PATH = "prompts/roles.json"

# Variables globales
history = []
modelo_actual = None
tipo_actual = None
personalidad_cfg = None
rol_cfg = None

# Cargar configuraciones
def cargar_modelos():
    with open(MODELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def cargar_personalidades():
    with open(PERSONALIDADES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def cargar_roles():
    with open(ROLES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# Selección desde menú
def seleccionar_modelo(modelos):
    print("\nSelecciona un modelo:")
    for clave, data in modelos.items():
        print(f"{data['nombre']:<20} {clave}")
    while True:
        eleccion = input("\nIngresa el número del modelo: ").strip()
        if eleccion in modelos:
            return modelos[eleccion]
        print("Opción inválida.")

def seleccionar_personalidad(personalidades):
    print("\nSelecciona una personalidad:")
    for i, p in enumerate(personalidades, start=1):
        print(f"{p['nombre']:<25} {i}")
    while True:
        index = input("\nIngresa el número de tu elección: ").strip()
        if index.isdigit() and 1 <= int(index) <= len(personalidades):
            return personalidades[int(index) - 1]
        print("Opción inválida.")

def seleccionar_rol(roles):
    print("\nSelecciona un rol profesional:")
    for i, r in enumerate(roles, start=1):
        print(f"{r['nombre']:<30} {i}")
    while True:
        index = input("\nIngresa el número del rol: ").strip()
        if index.isdigit() and 1 <= int(index) <= len(roles):
            return roles[int(index) - 1]
        print("Opción inválida.")

# Inicializar conversación con nueva configuración
def iniciar_conversacion(modelo_config, personalidad_config, rol_config):
    global history, modelo_actual, tipo_actual
    modelo_actual = modelo_config["modelo"]
    tipo_actual = modelo_config["tipo"]

    system_prompt = f"""{personalidad_config['saludo']}

Eres un asistente con el rol de {rol_config['nombre']}. 
{rol_config['descripcion']}
Estilo: {rol_config['estilo_rol']}
Especialidades: {', '.join(rol_config['ramas'])}

{personalidad_config['descripcion']}
"""

    history = [{"role": "system", "content": system_prompt}]
    print(f"\nBot: {personalidad_config['saludo']}")

# Enviar mensaje al modelo correcto
def enviar_mensaje(mensaje):
    history.append({"role": "user", "content": mensaje})

    if tipo_actual == "openai":
        respuesta = openai.chat.completions.create(
            model=modelo_actual,
            messages=history,
            temperature=0.5
        )
        reply = respuesta.choices[0].message.content

    elif tipo_actual == "ollama":
        payload = {
            "model": modelo_actual,
            "messages": history,
            "stream": False  # ⚠️ Forzamos respuesta en bloque
        }
        response = requests.post("http://localhost:11434/api/chat", json=payload)

        try:
            result = response.json()
            reply = result["message"]["content"]
        except Exception as e:
            print("Error al procesar la respuesta de Ollama:")
            print(response.text)
            reply = "[Error de formato en respuesta del modelo Ollama]"

    else:
        reply = "Error: tipo de modelo no soportado."

    history.append({"role": "assistant", "content": reply})
    return reply

# MAIN
if __name__ == "__main__":
    print("Chatbot iniciado. Escribe 'salir' para terminar, 'cambiar personalidad' o 'cambiar modelo' para modificar la conversación.")

    modelos = cargar_modelos()
    personalidades = cargar_personalidades()
    roles = cargar_roles()

    modelo_cfg = seleccionar_modelo(modelos)
    personalidad_cfg = seleccionar_personalidad(personalidades)
    rol_cfg = seleccionar_rol(roles)

    iniciar_conversacion(modelo_cfg, personalidad_cfg, rol_cfg)

    while True:
        user_input = input("\nTú: ").strip().lower()
        if user_input in ["salir", "exit"]:
            print("Hasta luego.")
            break
        elif user_input in ["cambiar personalidad", "switch"]:
            personalidad_cfg = seleccionar_personalidad(personalidades)
            iniciar_conversacion(modelo_cfg, personalidad_cfg, rol_cfg)
            continue
        elif user_input in ["cambiar modelo", "modelo"]:
            modelo_cfg = seleccionar_modelo(modelos)
            iniciar_conversacion(modelo_cfg, personalidad_cfg, rol_cfg)
            continue
        elif user_input in ["cambiar rol", "rol"]:
            rol_cfg = seleccionar_rol(roles)
            iniciar_conversacion(modelo_cfg, personalidad_cfg, rol_cfg)
            continue

        respuesta = enviar_mensaje(user_input)
        print(f"Bot: {respuesta}")

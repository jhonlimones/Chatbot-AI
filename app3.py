# Chatnot con memoria y personalidades

import openai
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Diccionario de personalidades (clave, (archivo, emoji + nombre))
personalidades = {
    "1": ("empresarial", "🧑 Empresarial"),
    "2": ("rudo", "💢 Rudo"),
    "3": ("amable", "💖 Amable"),
    "4": ("filosofico", "🧠 Filosófico"),
    "5": ("tecnico", "🧪 Técnico"),
    "6": ("salir", "❌ Salir")
}

# Cargar prompt
def cargar_personalidad(nombre_personalidad):
    ruta = f"prompts/{nombre_personalidad}.txt"
    try:
        with open(ruta, "r", encoding="utf-8") as archivo:
            return archivo.read()
    except FileNotFoundError:
        raise ValueError(f"No se encontró la personalidad '{nombre_personalidad}' en {ruta}")

# Mostrar menú alineado
def mostrar_menu():
    print("\nSelecciona una personalidad:")
    for key in personalidades:
        nombre = personalidades[key][1]
        print(f"{nombre:<20} {key}")

# Elegir opción válida
def seleccionar_personalidad():
    while True:
        mostrar_menu()
        eleccion = input("\nIngresa el número de tu elección: ").strip()
        if eleccion == "6":
            print("\nSaliendo del chatbot. ¡Hasta luego!")
            exit()
        if eleccion in personalidades and eleccion != "6":
            return personalidades[eleccion][0]

# Inicializar conversación
def iniciar_conversacion():
    global history
    personalidad = seleccionar_personalidad()
    system_prompt = cargar_personalidad(personalidad)
    history = [{"role": "system", "content": system_prompt}]
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
        temperature=0.5
    )
    intro = response.choices[0].message.content
    history.append({"role": "assistant", "content": intro})
    print(f"\nBot: {intro}")

# Chat con memoria
def chat_with_memory(user_input):
    history.append({"role": "user", "content": user_input})
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
        temperature=0.5
    )
    bot_reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": bot_reply})
    return bot_reply

# --- MAIN ---
if __name__ == "__main__":
    print("Chatbot con memoria iniciado. Escribe 'salir' para terminar o 'cambiar personalidad' en cualquier momento.")

    iniciar_conversacion()

    while True:
        user_input = input("\nTú: ").strip().lower()
        if user_input in ["salir", "exit"]:
            print("Hasta luego.")
            break
        elif user_input in ["cambiar personalidad", "switch"]:
            iniciar_conversacion()
            continue

        respuesta = chat_with_memory(user_input)
        print(f"Bot: {respuesta}")

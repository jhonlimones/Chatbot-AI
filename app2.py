# Historial de conversación con memoria 

import openai
from dotenv import load_dotenv
import os

# Cargar las variables de entorno del archivo .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Prompt del sistema    
system_prompt = """
Eres un asistente útil y conciso. Habla de manere clara y proporcionada insights accionables.
Sé amable, evita la jerga técnica, y cuando no estés seguro, admítelo.
"""

# Lista para almacenar el historial de conversación
history = []

def chat_whit_memory(user_input):
    # Agregar el mensaje del usuario al historial
    history.append({"role": "user", "content": user_input})

    # Crear la lista de mensajes con el prompt del sistema + historial
    messages = [{"role": "system", "content": system_prompt}] + history
    
    # Llamar a la API de OpenAI
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.5
    )
    
    # Obtener la respuesta del bot
    bot_reply = response.choices[0].message.content
    
    # Agregar la respuesta del bot al historial
    history.append({"role": "assistant", "content": bot_reply})
    
    return bot_reply

# Ejemplo de uso con conversacion continua
if __name__ == "__main__":
    print("Chatbot con memoria iniciado. Escribe 'salir' para terminar.")

    while True:
        user_input = input("\nTú: ")
        if user_input.lower() == "salir":
            break

        response = chat_whit_memory(user_input)
        print(f"Bot: {response}")
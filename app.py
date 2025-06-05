# chatbot simple

import openai
from dotenv import load_dotenv
import os

# Cargar las variables de entorno del archivo .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Prompt del sis    
system_prompt = """
Eres un asistente útil y conciso. Habla de manere clara y proporcionada insights accionables.
Sé amable, evita la jerga técnica, y cuando no estés seguro, admítelo.
"""

def chat_whit_bot(user_input):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.5,
    )
    return response.choices[0].message.content

# Ejemplo de uso
if __name__ == "__main__":
    user_message = "Hola, ¿?Cómo estás?"
    response = chat_whit_bot(user_message)
    print(response)
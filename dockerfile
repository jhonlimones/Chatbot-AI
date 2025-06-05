# Imagen base con Python
FROM python:3.12.3

# Establecer directorio de trabajo
WORKDIR /app

# Copiar los archivos esenciales
COPY *.py .
COPY prompts/ prompts/
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Comando por defecto
CMD ["python", "app.py"]

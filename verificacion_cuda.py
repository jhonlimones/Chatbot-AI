"""
Script de verificación de entorno CUDA + PyTorch + GPU
-------------------------------------------------------
Este script comprueba si PyTorch fue instalado con soporte CUDA y si la GPU está disponible.
También imprime información relevante sobre la GPU y versión de CUDA en uso.
Ideal para validación antes de correr modelos de generación de imágenes.
"""

import torch
import platform
import os

def verificar_entorno_cuda():
    print("🔍 Verificando configuración de entorno PyTorch + CUDA...\n")

    # 1. Verificar sistema operativo
    print(f"🖥️  Sistema operativo: {platform.system()} {platform.release()}")
    print(f"💻 Arquitectura: {platform.machine()}")
    print(f"📦 Python versión: {platform.python_version()}")
    print(f"📂 Virtual env: {'Sí' if 'VIRTUAL_ENV' in os.environ else 'No'}")

    # 2. Verificar disponibilidad de CUDA
    print("\n🚀 CUDA disponible en PyTorch:", torch.cuda.is_available())

    if torch.cuda.is_available():
        # 3. Información detallada de la GPU
        num_gpus = torch.cuda.device_count()
        print(f"🔢 GPUs detectadas: {num_gpus}")

        for i in range(num_gpus):
            nombre = torch.cuda.get_device_name(i)
            memoria_total = round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2)
            capacidad = torch.cuda.get_device_capability(i)
            print(f"   - GPU {i}: {nombre}")
            print(f"     🧠 Memoria: {memoria_total} GB")
            print(f"     📊 Capacidad CUDA: {capacidad}")
        
        # 4. Verificar versión CUDA de PyTorch
        print(f"\n🔧 Versión CUDA (compilada en PyTorch): {torch.version.cuda}")
        print(f"🔧 Versión cuDNN: {torch.backends.cudnn.version()}")
    else:
        print("❌ Tu instalación actual de PyTorch no tiene soporte CUDA.")
        print("   ➤ Reinstala PyTorch con CUDA desde: https://pytorch.org/get-started/locally/")

if __name__ == "__main__":
    verificar_entorno_cuda()

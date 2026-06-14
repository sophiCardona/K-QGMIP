#!/usr/bin/env python3
"""
Script para generar archivos TPM (samples) de diferentes tamaños.

Uso:
    python3 generar_samples.py
"""

from src.controllers.manager import Manager
from pathlib import Path

def generar_sample(n_bits: int):
    """Genera un archivo de sample TPM para n bits."""
    estado_inicial = "1" + "0" * (n_bits - 1)
    print(f"\n{'='*60}")
    print(f"Generando sample para N{n_bits}A ({n_bits} bits)")
    print(f"{'='*60}")
    
    try:
        gestor = Manager(estado_inicial=estado_inicial)
        print(f"✓ Manager creado para {n_bits} bits")
        print(f"✓ Estado inicial: {estado_inicial}")
        
        # Generar la red (TPM) realmente
        resultado = gestor.generar_red(dimensiones=n_bits, datos_discretos=True)
        print(f"✓ Sample generado: {resultado}")
        
    except Exception as e:
        print(f"✗ Error al generar N{n_bits}A: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Genera todos los samples necesarios."""
    tamaños = [20, 22, 25]
    
    print("Iniciando generación de samples...")
    for tamaño in tamaños:
        generar_sample(tamaño)
    
    print(f"\n{'='*60}")
    print("✓ Generación completada")
    print("Los archivos se encuentran en: GeoMIP/data/samples/")
    print("Archivos generados:")
    for tamaño in tamaños:
        print(f"  - N{tamaño}A.csv")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()


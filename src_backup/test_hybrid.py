#!/usr/bin/env python3
"""
Script para testar o quebrador híbrido com o quebrador otimizado para matriz 3x3.
"""

import os
import time
import numpy as np
from typing import List, Tuple

# Importar os diferentes quebradores
try:
    from hill_cipher_breaker import HillCipherBreaker as BasicBreaker
    from hill_cipher_breaker_optimized import HillCipherBreaker as OptimizedBreaker
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.hill_cipher_breaker import HillCipherBreaker as BasicBreaker
    from src.hill_cipher_breaker_optimized import HillCipherBreaker as OptimizedBreaker

def test_hybrid_approach(known_dir: str = "textos_conhecidos"):
    """
    Testa a abordagem híbrida: quebrador básico para 2x2 e otimizado para 3x3.
    
    Args:
        known_dir: Diretório com textos conhecidos
    """
    # Verificar se o diretório existe
    if not os.path.exists(known_dir):
        print(f"Erro: Diretório {known_dir} não encontrado.")
        return
    
    # Criar instâncias dos quebradores
    basic_breaker = BasicBreaker()
    optimized_breaker = OptimizedBreaker()
    
    # Processar matriz 2x2
    print("\n=== Processando matriz 2x2 com quebrador BÁSICO ===")
    process_matrix(2, basic_breaker, known_dir)
    
    # Processar matriz 3x3
    print("\n=== Processando matriz 3x3 com quebrador OTIMIZADO ===")
    process_matrix(3, optimized_breaker, known_dir)

def process_matrix(size: int, breaker, known_dir: str):
    """
    Processa uma matriz de tamanho específico com o quebrador fornecido.
    
    Args:
        size: Tamanho da matriz
        breaker: Instância do quebrador
        known_dir: Diretório com textos conhecidos
    """
    # Caminho para o texto cifrado
    cipher_path = os.path.join(known_dir, "Cifrado", "Hill", f"Grupo02_{size}_texto_cifrado.txt")
    
    if not os.path.exists(cipher_path):
        print(f"Erro: Arquivo {cipher_path} não encontrado.")
        return
    
    # Encontrar o texto original correspondente
    original_text_path = None
    for text_file in os.listdir(os.path.join(known_dir, "textos")):
        if text_file.endswith(".txt"):
            original_text_path = os.path.join(known_dir, "textos", text_file)
            print(f"Usando texto original: {original_text_path}")
            break
    
    # Ler texto cifrado
    with open(cipher_path, 'r') as f:
        ciphertext = f.read().strip()
    
    # Quebrar cifra
    print(f"Quebrando cifra {size}x{size}...")
    start_time = time.time()
    
    if isinstance(breaker, OptimizedBreaker):
        results = breaker.break_cipher(ciphertext, size, known_text_path=original_text_path)
    else:
        results = breaker.break_cipher(ciphertext, size)
    
    elapsed_time = time.time() - start_time
    
    # Gerar relatório
    if results:
        if isinstance(breaker, OptimizedBreaker):
            report = breaker.generate_report(results, ciphertext, size, known_text_path=original_text_path)
        else:
            report = breaker.generate_report(results, ciphertext, size)
        
        print(report)
        print(f"Tempo de execução: {elapsed_time:.2f} segundos")
        
        # Salvar relatório
        report_dir = f"relatorios/hybrid_test/hill_{size}x{size}"
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "relatorio.txt")
        with open(report_path, 'w') as f:
            f.write(report)
            f.write(f"\n\nTempo de execução: {elapsed_time:.2f} segundos")
        print(f"Relatório salvo em {report_path}")
    else:
        print("Nenhum resultado encontrado.")

if __name__ == "__main__":
    test_hybrid_approach()

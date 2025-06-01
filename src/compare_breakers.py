#!/usr/bin/env python3
"""
Script para comparar os resultados do quebrador básico e otimizado para a matriz 2x2.
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

def compare_breakers_2x2(known_dir: str = "textos_conhecidos"):
    """
    Compara os resultados do quebrador básico e otimizado para a matriz 2x2.
    
    Args:
        known_dir: Diretório com textos conhecidos
    """
    # Verificar se o diretório existe
    if not os.path.exists(known_dir):
        print(f"Erro: Diretório {known_dir} não encontrado.")
        return
    
    # Caminho para o texto cifrado
    cipher_path = os.path.join(known_dir, "Cifrado", "Hill", "Grupo02_2_texto_cifrado.txt")
    
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
    
    # Criar instâncias dos quebradores
    basic_breaker = BasicBreaker()
    optimized_breaker = OptimizedBreaker()
    
    # Quebrar cifra com o quebrador básico
    print("\n=== Quebrando cifra 2x2 com o quebrador BÁSICO ===")
    start_time = time.time()
    basic_results = basic_breaker.break_cipher(ciphertext, 2)
    basic_time = time.time() - start_time
    
    # Quebrar cifra com o quebrador otimizado
    print("\n=== Quebrando cifra 2x2 com o quebrador OTIMIZADO ===")
    start_time = time.time()
    optimized_results = optimized_breaker.break_cipher(ciphertext, 2, known_text_path=original_text_path)
    optimized_time = time.time() - start_time
    
    # Comparar resultados
    print("\n=== COMPARAÇÃO DE RESULTADOS ===")
    print(f"Tempo do quebrador básico: {basic_time:.2f} segundos")
    print(f"Tempo do quebrador otimizado: {optimized_time:.2f} segundos")
    print(f"Diferença de tempo: {optimized_time - basic_time:.2f} segundos")
    
    # Comparar as 5 melhores matrizes
    print("\nMelhores 5 matrizes do quebrador BÁSICO:")
    for i, (matrix, text, score) in enumerate(basic_results[:5], 1):
        print(f"{i}. Matriz: {matrix.tolist()}, Score: {score:.4f}")
        print(f"   Texto: {text[:50]}..." if len(text) > 50 else text)
    
    print("\nMelhores 5 matrizes do quebrador OTIMIZADO:")
    for i, (matrix, text, score) in enumerate(optimized_results[:5], 1):
        print(f"{i}. Matriz: {matrix.tolist()}, Score: {score:.4f}")
        print(f"   Texto: {text[:50]}..." if len(text) > 50 else text)
    
    # Verificar se as matrizes são iguais
    print("\nVerificação de matrizes iguais:")
    basic_matrices = [tuple(map(tuple, matrix.tolist())) for matrix, _, _ in basic_results[:20]]
    optimized_matrices = [tuple(map(tuple, matrix.tolist())) for matrix, _, _ in optimized_results[:20]]
    
    common_matrices = set(basic_matrices).intersection(set(optimized_matrices))
    print(f"Número de matrizes em comum entre os 20 melhores resultados: {len(common_matrices)}")
    
    if common_matrices:
        print("Matrizes em comum:")
        for matrix in common_matrices:
            basic_idx = basic_matrices.index(matrix)
            optimized_idx = optimized_matrices.index(matrix)
            print(f"Matriz: {matrix}")
            print(f"  Posição no quebrador básico: {basic_idx + 1}")
            print(f"  Posição no quebrador otimizado: {optimized_idx + 1}")
    
    # Salvar relatório
    report_dir = "relatorios/comparacao"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "comparacao_2x2.txt")
    
    with open(report_path, 'w') as f:
        f.write("=== COMPARAÇÃO DE QUEBRADORES PARA MATRIZ 2x2 ===\n\n")
        f.write(f"Tempo do quebrador básico: {basic_time:.2f} segundos\n")
        f.write(f"Tempo do quebrador otimizado: {optimized_time:.2f} segundos\n")
        f.write(f"Diferença de tempo: {optimized_time - basic_time:.2f} segundos\n\n")
        
        f.write("Melhores 5 matrizes do quebrador BÁSICO:\n")
        for i, (matrix, text, score) in enumerate(basic_results[:5], 1):
            f.write(f"{i}. Matriz: {matrix.tolist()}, Score: {score:.4f}\n")
            f.write(f"   Texto: {text[:100]}...\n\n" if len(text) > 100 else f"   Texto: {text}\n\n")
        
        f.write("Melhores 5 matrizes do quebrador OTIMIZADO:\n")
        for i, (matrix, text, score) in enumerate(optimized_results[:5], 1):
            f.write(f"{i}. Matriz: {matrix.tolist()}, Score: {score:.4f}\n")
            f.write(f"   Texto: {text[:100]}...\n\n" if len(text) > 100 else f"   Texto: {text}\n\n")
        
        f.write(f"Número de matrizes em comum entre os 20 melhores resultados: {len(common_matrices)}\n\n")
        
        if common_matrices:
            f.write("Matrizes em comum:\n")
            for matrix in common_matrices:
                basic_idx = basic_matrices.index(matrix)
                optimized_idx = optimized_matrices.index(matrix)
                f.write(f"Matriz: {matrix}\n")
                f.write(f"  Posição no quebrador básico: {basic_idx + 1}\n")
                f.write(f"  Posição no quebrador otimizado: {optimized_idx + 1}\n\n")
    
    print(f"\nRelatório salvo em {report_path}")

def compare_breakers_3x3(known_dir: str = "textos_conhecidos"):
    """
    Compara os resultados do quebrador básico e otimizado para a matriz 3x3.
    
    Args:
        known_dir: Diretório com textos conhecidos
    """
    # Verificar se o diretório existe
    if not os.path.exists(known_dir):
        print(f"Erro: Diretório {known_dir} não encontrado.")
        return
    
    # Caminho para o texto cifrado
    cipher_path = os.path.join(known_dir, "Cifrado", "Hill", "Grupo02_3_texto_cifrado.txt")
    
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
    
    # Criar instâncias dos quebradores
    basic_breaker = BasicBreaker()
    optimized_breaker = OptimizedBreaker()
    
    # Quebrar cifra com o quebrador básico
    print("\n=== Quebrando cifra 3x3 com o quebrador BÁSICO ===")
    start_time = time.time()
    basic_results = basic_breaker.break_cipher(ciphertext, 3)
    basic_time = time.time() - start_time
    
    # Quebrar cifra com o quebrador otimizado
    print("\n=== Quebrando cifra 3x3 com o quebrador OTIMIZADO ===")
    start_time = time.time()
    optimized_results = optimized_breaker.break_cipher(ciphertext, 3, known_text_path=original_text_path)
    optimized_time = time.time() - start_time
    
    # Comparar resultados
    print("\n=== COMPARAÇÃO DE RESULTADOS ===")
    print(f"Tempo do quebrador básico: {basic_time:.2f} segundos")
    print(f"Tempo do quebrador otimizado: {optimized_time:.2f} segundos")
    print(f"Diferença de tempo: {optimized_time - basic_time:.2f} segundos")
    
    # Comparar as 5 melhores matrizes
    print("\nMelhores 5 matrizes do quebrador BÁSICO:")
    for i, (matrix, text, score) in enumerate(basic_results[:5], 1):
        print(f"{i}. Matriz: {matrix.tolist()}, Score: {score:.4f}")
        print(f"   Texto: {text[:50]}..." if len(text) > 50 else text)
    
    print("\nMelhores 5 matrizes do quebrador OTIMIZADO:")
    for i, (matrix, text, score) in enumerate(optimized_results[:5], 1):
        print(f"{i}. Matriz: {matrix.tolist()}, Score: {score:.4f}")
        print(f"   Texto: {text[:50]}..." if len(text) > 50 else text)
    
    # Verificar se as matrizes são iguais
    print("\nVerificação de matrizes iguais:")
    basic_matrices = [tuple(map(tuple, matrix.tolist())) for matrix, _, _ in basic_results[:20]]
    optimized_matrices = [tuple(map(tuple, matrix.tolist())) for matrix, _, _ in optimized_results[:20]]
    
    common_matrices = set(basic_matrices).intersection(set(optimized_matrices))
    print(f"Número de matrizes em comum entre os 20 melhores resultados: {len(common_matrices)}")
    
    if common_matrices:
        print("Matrizes em comum:")
        for matrix in common_matrices:
            basic_idx = basic_matrices.index(matrix)
            optimized_idx = optimized_matrices.index(matrix)
            print(f"Matriz: {matrix}")
            print(f"  Posição no quebrador básico: {basic_idx + 1}")
            print(f"  Posição no quebrador otimizado: {optimized_idx + 1}")
    
    # Salvar relatório
    report_dir = "relatorios/comparacao"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "comparacao_3x3.txt")
    
    with open(report_path, 'w') as f:
        f.write("=== COMPARAÇÃO DE QUEBRADORES PARA MATRIZ 3x3 ===\n\n")
        f.write(f"Tempo do quebrador básico: {basic_time:.2f} segundos\n")
        f.write(f"Tempo do quebrador otimizado: {optimized_time:.2f} segundos\n")
        f.write(f"Diferença de tempo: {optimized_time - basic_time:.2f} segundos\n\n")
        
        f.write("Melhores 5 matrizes do quebrador BÁSICO:\n")
        for i, (matrix, text, score) in enumerate(basic_results[:5], 1):
            f.write(f"{i}. Matriz: {matrix.tolist()}, Score: {score:.4f}\n")
            f.write(f"   Texto: {text[:100]}...\n\n" if len(text) > 100 else f"   Texto: {text}\n\n")
        
        f.write("Melhores 5 matrizes do quebrador OTIMIZADO:\n")
        for i, (matrix, text, score) in enumerate(optimized_results[:5], 1):
            f.write(f"{i}. Matriz: {matrix.tolist()}, Score: {score:.4f}\n")
            f.write(f"   Texto: {text[:100]}...\n\n" if len(text) > 100 else f"   Texto: {text}\n\n")
        
        f.write(f"Número de matrizes em comum entre os 20 melhores resultados: {len(common_matrices)}\n\n")
        
        if common_matrices:
            f.write("Matrizes em comum:\n")
            for matrix in common_matrices:
                basic_idx = basic_matrices.index(matrix)
                optimized_idx = optimized_matrices.index(matrix)
                f.write(f"Matriz: {matrix}\n")
                f.write(f"  Posição no quebrador básico: {basic_idx + 1}\n")
                f.write(f"  Posição no quebrador otimizado: {optimized_idx + 1}\n\n")
    
    print(f"\nRelatório salvo em {report_path}")

if __name__ == "__main__":
    print("=== Comparação de Quebradores de Cifra de Hill ===")
    print("1. Comparar para matriz 2x2")
    print("2. Comparar para matriz 3x3")
    print("3. Comparar para ambas as matrizes")
    
    choice = input("Escolha uma opção (1-3): ")
    
    if choice == "1":
        compare_breakers_2x2()
    elif choice == "2":
        compare_breakers_3x3()
    elif choice == "3":
        compare_breakers_2x2()
        compare_breakers_3x3()
    else:
        print("Opção inválida.")

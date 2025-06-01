#!/usr/bin/env python3
"""
Script para debugar o desempenho do quebrador de cifra de Hill.
"""

import os
import time
import cProfile
import pstats
from pstats import SortKey
import sys

# Importar funções do quebrador de cifra
from hill_cipher_breaker_advanced import (
    HillCipherBreaker, 
    generate_invertible_matrices,
    brute_force_hill_parallel,
    process_chunk
)

def debug_matrix_generation():
    """Debugar a geração de matrizes inversíveis."""
    print("Debugando geração de matrizes inversíveis 2x2...")
    
    start_time = time.time()
    matrices = generate_invertible_matrices(2, limit=1000)  # Limitar para teste
    elapsed_time = time.time() - start_time
    
    print(f"Geradas {len(matrices)} matrizes em {elapsed_time:.2f} segundos")
    print(f"Tempo médio por matriz: {elapsed_time / len(matrices):.6f} segundos")
    
    return matrices

def debug_brute_force(ciphertext, num_processes=None):
    """Debugar a força bruta paralela."""
    print("Debugando força bruta paralela para matriz 2x2...")
    
    start_time = time.time()
    results = brute_force_hill_parallel(ciphertext, 2, None, num_processes)
    elapsed_time = time.time() - start_time
    
    print(f"Processadas {len(results)} matrizes em {elapsed_time:.2f} segundos")
    if results:
        print(f"Melhor resultado: {results[0][1][:50]}...")
    
    return results

def debug_process_chunk(matrices, ciphertext):
    """Debugar o processamento de um chunk de matrizes."""
    print("Debugando processamento de chunk...")
    
    start_time = time.time()
    results = process_chunk((matrices[:100], ciphertext, None))  # Testar com 100 matrizes
    elapsed_time = time.time() - start_time
    
    print(f"Processadas 100 matrizes em {elapsed_time:.2f} segundos")
    print(f"Tempo médio por matriz: {elapsed_time / 100:.6f} segundos")
    
    return results

def profile_hill_breaker():
    """Perfilar o quebrador de cifra de Hill."""
    print("Perfilando quebrador de cifra de Hill...")
    
    # Diretórios de textos
    known_dir = "textos_conhecidos"
    
    # Caminho para o texto cifrado
    cipher_path = os.path.join(known_dir, "Cifrado", "Hill", "Grupo02_2_texto_cifrado.txt")
    
    # Ler texto cifrado
    with open(cipher_path, 'r') as f:
        ciphertext = f.read().strip()
    
    # Criar instância do quebrador de cifra
    breaker = HillCipherBreaker()
    
    # Perfilar a quebra de cifra
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Limitar o número de matrizes para teste
    results = breaker.break_cipher(ciphertext, 2)
    
    profiler.disable()
    
    # Salvar resultados do perfil
    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)  # Mostrar as 20 funções que mais consomem tempo
    
    # Salvar estatísticas em arquivo
    stats.dump_stats("hill_cipher_profile.prof")
    print("Estatísticas de perfil salvas em 'hill_cipher_profile.prof'")
    
    return results

def main():
    """Função principal."""
    print("=== Debug do Quebrador de Cifra de Hill ===")
    
    # Diretórios de textos
    known_dir = "textos_conhecidos"
    
    # Caminho para o texto cifrado
    cipher_path = os.path.join(known_dir, "Cifrado", "Hill", "Grupo02_2_texto_cifrado.txt")
    
    # Verificar se o arquivo existe
    if not os.path.exists(cipher_path):
        print(f"Erro: Arquivo {cipher_path} não encontrado.")
        return
    
    # Ler texto cifrado
    with open(cipher_path, 'r') as f:
        ciphertext = f.read().strip()
    
    # Debugar geração de matrizes
    matrices = debug_matrix_generation()
    
    # Debugar processamento de chunk
    debug_process_chunk(matrices, ciphertext)
    
    # Debugar força bruta com número limitado de processos
    debug_brute_force(ciphertext, 2)
    
    # Perfilar o quebrador completo
    profile_hill_breaker()

if __name__ == "__main__":
    main()

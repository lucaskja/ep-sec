#!/usr/bin/env python3
"""
Hill Cipher Hybrid Breaker - Combina as abordagens básica e otimizada

Este script implementa uma abordagem híbrida para quebrar a cifra de Hill:
- Usa o quebrador básico (rápido) para matrizes 2x2 e 3x3
- Usa o quebrador otimizado (mais preciso) para matrizes 4x4 e 5x5

Autor: Amazon Q
"""

import os
import time
import sys
import argparse
from typing import List, Tuple, Dict, Optional

# Importar os diferentes quebradores
try:
    from hill_cipher_breaker import HillCipherBreaker as BasicBreaker
    from hill_cipher_breaker_optimized import HillCipherBreaker as OptimizedBreaker
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.hill_cipher_breaker import HillCipherBreaker as BasicBreaker
    from src.hill_cipher_breaker_optimized import HillCipherBreaker as OptimizedBreaker

def process_all_ciphers(known_dir: str = "textos_conhecidos", 
                        unknown_dir: str = "textos_desconhecidos",
                        sizes: List[int] = [2, 3, 4, 5],
                        save_reports: bool = True) -> Dict:
    """
    Processa todas as cifras usando a abordagem híbrida.
    
    Args:
        known_dir: Diretório com textos conhecidos
        unknown_dir: Diretório com textos desconhecidos
        sizes: Tamanhos de matriz a processar
        save_reports: Se True, salva relatórios em arquivos
        
    Returns:
        Dicionário com resultados e estatísticas
    """
    results = {}
    
    # Verificar se os diretórios existem
    if not os.path.exists(known_dir) or not os.path.exists(unknown_dir):
        print(f"Erro: Diretórios {known_dir} ou {unknown_dir} não encontrados.")
        return results
    
    # Criar instâncias dos quebradores
    basic_breaker = BasicBreaker()
    optimized_breaker = OptimizedBreaker()
    
    print("=== Hill Cipher Hybrid Breaker ===")
    print("Usando abordagem híbrida:")
    print("- Quebrador básico para matrizes 2x2 e 3x3")
    print("- Quebrador otimizado para matrizes 4x4 e 5x5")
    
    # Processar textos conhecidos
    print("\n=== Processando textos conhecidos ===")
    for size in sizes:
        cipher_path = os.path.join(known_dir, "Cifrado", "Hill", f"Grupo02_{size}_texto_cifrado.txt")
        
        if not os.path.exists(cipher_path):
            print(f"Arquivo {cipher_path} não encontrado. Pulando...")
            continue
        
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
        
        print(f"\n--- Quebrando cifra {size}x{size} (texto conhecido) ---")
        start_time = time.time()
        
        # Escolher o quebrador apropriado
        if size in [2, 3]:
            print(f"Usando quebrador básico para matriz {size}x{size}...")
            results_list = basic_breaker.break_cipher(ciphertext, size)
        else:
            print(f"Usando quebrador otimizado para matriz {size}x{size}...")
            try:
                # Usar diretamente o quebrador otimizado
                results_list = optimized_breaker.break_cipher(ciphertext, size)
            except Exception as e:
                print(f"Erro nas técnicas avançadas: {e}")
                results_list = []
        
        elapsed_time = time.time() - start_time
        
        # Gerar relatório
        if results_list:
            if size in [2, 3]:
                report = basic_breaker.generate_report(results_list, ciphertext, size)
            else:
                report = optimized_breaker.generate_report(results_list, ciphertext, size)
            
            print(report)
            print(f"Tempo de execução: {elapsed_time:.2f} segundos")
            
            # Salvar relatório
            if save_reports:
                report_dir = f"relatorios/hibrido/conhecidos/hill_{size}x{size}"
                os.makedirs(report_dir, exist_ok=True)
                report_path = os.path.join(report_dir, "relatorio.txt")
                with open(report_path, 'w') as f:
                    f.write(report)
                    f.write(f"\n\nTempo de execução: {elapsed_time:.2f} segundos")
                print(f"Relatório salvo em {report_path}")
            
            # Armazenar resultados
            results[f"known_{size}x{size}"] = {
                "best_matrix": results_list[0][0].tolist() if results_list else None,
                "best_text": results_list[0][1][:100] if results_list else None,
                "best_score": results_list[0][2] if results_list else None,
                "time": elapsed_time
            }
        else:
            print("Nenhum resultado encontrado.")
            results[f"known_{size}x{size}"] = {"error": "Nenhum resultado encontrado"}
    
    # Processar textos desconhecidos
    print("\n=== Processando textos desconhecidos ===")
    for size in sizes:
        cipher_path = os.path.join(unknown_dir, "Cifrado", "Hill", f"Grupo02_{size}_texto_cifrado.txt")
        
        if not os.path.exists(cipher_path):
            print(f"Arquivo {cipher_path} não encontrado. Pulando...")
            continue
        
        # Ler texto cifrado
        with open(cipher_path, 'r') as f:
            ciphertext = f.read().strip()
        
        print(f"\n--- Quebrando cifra {size}x{size} (texto desconhecido) ---")
        start_time = time.time()
        
        # Escolher o quebrador apropriado
        if size in [2, 3]:
            print(f"Usando quebrador básico para matriz {size}x{size}...")
            results_list = basic_breaker.break_cipher(ciphertext, size)
        else:
            print(f"Usando quebrador otimizado para matriz {size}x{size}...")
            try:
                # Usar diretamente o quebrador otimizado
                results_list = optimized_breaker.break_cipher(ciphertext, size)
            except Exception as e:
                print(f"Erro nas técnicas avançadas: {e}")
                results_list = []
        
        elapsed_time = time.time() - start_time
        
        # Gerar relatório
        if results_list:
            if size in [2, 3]:
                report = basic_breaker.generate_report(results_list, ciphertext, size)
            else:
                report = optimized_breaker.generate_report(results_list, ciphertext, size)
            
            print(report)
            print(f"Tempo de execução: {elapsed_time:.2f} segundos")
            
            # Salvar relatório
            if save_reports:
                report_dir = f"relatorios/hibrido/desconhecidos/hill_{size}x{size}"
                os.makedirs(report_dir, exist_ok=True)
                report_path = os.path.join(report_dir, "relatorio.txt")
                with open(report_path, 'w') as f:
                    f.write(report)
                    f.write(f"\n\nTempo de execução: {elapsed_time:.2f} segundos")
                print(f"Relatório salvo em {report_path}")
            
            # Armazenar resultados
            results[f"unknown_{size}x{size}"] = {
                "best_matrix": results_list[0][0].tolist() if results_list else None,
                "best_text": results_list[0][1][:100] if results_list else None,
                "best_score": results_list[0][2] if results_list else None,
                "time": elapsed_time
            }
        else:
            print("Nenhum resultado encontrado.")
            results[f"unknown_{size}x{size}"] = {"error": "Nenhum resultado encontrado"}
    
    # Salvar resumo dos resultados
    if save_reports:
        summary_path = "relatorios/hibrido/resumo.txt"
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w') as f:
            f.write("=== RESUMO DOS RESULTADOS ===\n\n")
            
            f.write("Textos Conhecidos:\n")
            for size in sizes:
                key = f"known_{size}x{size}"
                if key in results:
                    if "error" in results[key]:
                        f.write(f"- Matriz {size}x{size}: {results[key]['error']}\n")
                    else:
                        f.write(f"- Matriz {size}x{size}: Score {results[key]['best_score']:.4f}, Tempo {results[key]['time']:.2f}s\n")
            
            f.write("\nTextos Desconhecidos:\n")
            for size in sizes:
                key = f"unknown_{size}x{size}"
                if key in results:
                    if "error" in results[key]:
                        f.write(f"- Matriz {size}x{size}: {results[key]['error']}\n")
                    else:
                        f.write(f"- Matriz {size}x{size}: Score {results[key]['best_score']:.4f}, Tempo {results[key]['time']:.2f}s\n")
        
        print(f"\nResumo salvo em {summary_path}")
    
    return results

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="Hill Cipher Hybrid Breaker")
    parser.add_argument("--known-dir", default="textos_conhecidos", help="Diretório com textos conhecidos")
    parser.add_argument("--unknown-dir", default="textos_desconhecidos", help="Diretório com textos desconhecidos")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4, 5], help="Tamanhos de matriz a processar")
    parser.add_argument("--no-save", action="store_true", help="Não salvar relatórios em arquivos")
    
    args = parser.parse_args()
    
    # Processar todas as cifras
    process_all_ciphers(
        known_dir=args.known_dir,
        unknown_dir=args.unknown_dir,
        sizes=args.sizes,
        save_reports=not args.no_save
    )

if __name__ == "__main__":
    main()

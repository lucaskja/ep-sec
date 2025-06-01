#!/usr/bin/env python3
"""
Script para testar o quebrador de cifra de Hill otimizado.
"""

import os
import time
from hill_cipher_breaker_optimized import HillCipherBreaker

def test_single_cipher(size):
    """Testa a quebra de uma única cifra."""
    print(f"\n=== Testando quebra de cifra {size}x{size} ===")
    
    # Diretórios de textos
    known_dir = "textos_conhecidos"
    
    # Verificar se o diretório existe
    if not os.path.exists(known_dir):
        print(f"Erro: Diretório {known_dir} não encontrado.")
        return
    
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
    
    # Criar instância do quebrador de cifra
    breaker = HillCipherBreaker()
    
    # Quebrar cifra
    print(f"Quebrando cifra {size}x{size}...")
    start_time = time.time()
    results = breaker.break_cipher(ciphertext, size, known_text_path=original_text_path)
    elapsed_time = time.time() - start_time
    
    # Gerar relatório
    if results:
        report = breaker.generate_report(results, ciphertext, size, known_text_path=original_text_path)
        print(report)
        print(f"Tempo de execução: {elapsed_time:.2f} segundos")
        
        # Salvar relatório
        report_path = f"relatorios/otimizado/conhecidos/hill_{size}x{size}/relatorio_teste.txt"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)
            f.write(f"\n\nTempo de execução: {elapsed_time:.2f} segundos")
        print(f"Relatório salvo em {report_path}")
    else:
        print("Nenhum resultado encontrado.")

if __name__ == "__main__":
    # Testar cifra 2x2
    test_single_cipher(2)

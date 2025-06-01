#!/usr/bin/env python3
"""
Script para testar o quebrador de cifra de Hill avançado.
"""

import os
import time
from hill_cipher_breaker_advanced import HillCipherBreaker

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
    
    # Ler texto cifrado
    with open(cipher_path, 'r') as f:
        ciphertext = f.read().strip()
    
    # Criar instância do quebrador de cifra
    breaker = HillCipherBreaker()
    
    # Quebrar cifra
    print(f"Quebrando cifra {size}x{size}...")
    start_time = time.time()
    results = breaker.break_cipher(ciphertext, size)
    elapsed_time = time.time() - start_time
    
    # Gerar relatório
    if results:
        report = breaker.generate_report(results, ciphertext, size)
        print(report)
        print(f"Tempo de execução: {elapsed_time:.2f} segundos")
        
        # Salvar relatório
        report_path = f"relatorios/avancado/conhecidos/hill_{size}x{size}/relatorio_teste.txt"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)
            f.write(f"\n\nTempo de execução: {elapsed_time:.2f} segundos")
        print(f"Relatório salvo em {report_path}")
    else:
        print("Nenhum resultado encontrado.")

if __name__ == "__main__":
    # Testar cifra 2x2 com texto conhecido
    size = 2
    
    # Diretórios de textos
    known_dir = "textos_conhecidos"
    
    # Caminho para o texto cifrado
    cipher_path = os.path.join(known_dir, "Cifrado", "Hill", f"Grupo02_{size}_texto_cifrado.txt")
    
    # Texto conhecido (exemplo)
    known_plaintext = "NTAOPARANAOTERQUEENTRARNUMALUTACORPORALCOMMINHAMAEVOCETEVEQUESETRANCARNOBANHEIROEPASSOUALGUMTEMPOOUV"
    
    # Ler texto cifrado
    with open(cipher_path, 'r') as f:
        ciphertext = f.read().strip()
    
    # Criar instância do quebrador de cifra
    breaker = HillCipherBreaker()
    
    # Quebrar cifra com texto conhecido
    print(f"\n=== Testando quebra de cifra {size}x{size} com texto conhecido ===")
    
    # Verificar se o texto conhecido é válido para o ataque
    # Precisamos garantir que a matriz de texto claro seja inversível
    # Para isso, vamos ajustar o texto conhecido
    
    # Converter para números
    p_nums = [ord(c) - ord('A') for c in known_plaintext.upper() if c.isalpha()]
    
    # Garantir que temos blocos completos
    if len(p_nums) % size != 0:
        p_nums = p_nums[:-(len(p_nums) % size)]
    
    # Verificar se temos blocos suficientes
    if len(p_nums) < size * size:
        print(f"Texto conhecido muito curto. Precisamos de pelo menos {size * size} caracteres.")
    else:
        # Tentar diferentes posições iniciais para encontrar uma matriz inversível
        for start_pos in range(0, len(p_nums) - size * size + 1, size):
            test_plaintext = known_plaintext[start_pos:start_pos + size * size]
            try:
                from hill_cipher_breaker_advanced import text_to_numbers, is_invertible_matrix
                import numpy as np
                
                # Criar matriz de texto claro
                p_blocks = []
                test_nums = text_to_numbers(test_plaintext)
                for i in range(0, len(test_nums), size):
                    p_blocks.append(test_nums[i:i+size])
                P = np.array(p_blocks).T
                
                # Verificar se é inversível
                if is_invertible_matrix(P):
                    print(f"Encontrada matriz inversível na posição {start_pos}")
                    known_plaintext = test_plaintext
                    break
            except:
                continue
        else:
            print("Não foi possível encontrar uma matriz inversível no texto conhecido.")
            known_plaintext = None
    
    if known_plaintext:
        start_time = time.time()
        results = breaker.break_cipher(ciphertext, size, known_plaintext)
        elapsed_time = time.time() - start_time
        
        # Gerar relatório
        if results:
            report = breaker.generate_report(results, ciphertext, size)
            print(report)
            print(f"Tempo de execução: {elapsed_time:.2f} segundos")
            
            # Salvar relatório
            report_path = f"relatorio_test_known_hill_{size}x{size}.txt"
            with open(report_path, 'w') as f:
                f.write(report)
                f.write(f"\n\nTempo de execução: {elapsed_time:.2f} segundos")
            print(f"Relatório salvo em {report_path}")
    else:
        print("Usando força bruta em vez de ataque com texto conhecido.")
        start_time = time.time()
        results = breaker.break_cipher(ciphertext, size)
        elapsed_time = time.time() - start_time
        
        # Gerar relatório
        if results:
            report = breaker.generate_report(results, ciphertext, size)
            print(report)
            print(f"Tempo de execução: {elapsed_time:.2f} segundos")

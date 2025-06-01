#!/usr/bin/env python3
"""
Hill Cipher Breaker - Programa para decifrar a Cifra de Hill

Este programa implementa diversos métodos para quebrar a cifra de Hill,
incluindo ataque com texto conhecido, análise estatística, força bruta
e detecção de padrões.

Autor: Amazon Q
"""

import numpy as np
import os
import re
import math
import itertools
from collections import Counter
import time
from typing import List, Dict, Tuple, Optional, Union, Set

# Constantes
ALPHABET_SIZE = 26
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
LETTER_TO_NUM = {letter: idx for idx, letter in enumerate(ALPHABET)}
NUM_TO_LETTER = {idx: letter for idx, letter in enumerate(ALPHABET)}

# Funções auxiliares para operações matriciais
def mod_inverse(a: int, m: int = ALPHABET_SIZE) -> int:
    """
    Calcula o inverso modular de a mod m.
    
    Args:
        a: Número para calcular o inverso
        m: Módulo (padrão: 26)
        
    Returns:
        Inverso modular de a mod m
    
    Raises:
        ValueError: Se o inverso não existir
    """
    for i in range(1, m):
        if (a * i) % m == 1:
            return i
    raise ValueError(f"O inverso modular de {a} mod {m} não existe")

def matrix_mod_inverse(matrix: np.ndarray, mod: int = ALPHABET_SIZE) -> np.ndarray:
    """
    Calcula a inversa de uma matriz mod 26.
    
    Args:
        matrix: Matriz quadrada para inverter
        mod: Módulo (padrão: 26)
        
    Returns:
        Matriz inversa mod 26
        
    Raises:
        ValueError: Se a matriz não for inversível mod 26
    """
    n = matrix.shape[0]
    
    # Calcular determinante e verificar se é inversível
    det = int(round(np.linalg.det(matrix))) % mod
    
    # Verificar se o determinante tem inverso modular
    try:
        det_inv = mod_inverse(det, mod)
    except ValueError:
        raise ValueError("A matriz não é inversível mod 26")
    
    # Para matriz 2x2, usar fórmula direta
    if n == 2:
        adj = np.array([
            [matrix[1, 1], -matrix[0, 1]],
            [-matrix[1, 0], matrix[0, 0]]
        ]) % mod
        return (det_inv * adj) % mod
    
    # Para matrizes maiores, calcular a adjunta
    adj = np.zeros_like(matrix)
    for i in range(n):
        for j in range(n):
            # Cofator
            minor = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
            cofactor = int(round(np.linalg.det(minor))) % mod
            if (i + j) % 2 == 1:
                cofactor = (-cofactor) % mod
            adj[j, i] = cofactor  # Transposta da matriz de cofatores
    
    # Multiplicar pelo inverso do determinante
    return (det_inv * adj) % mod

def gcd(a: int, b: int) -> int:
    """Calcula o máximo divisor comum entre a e b."""
    while b:
        a, b = b, a % b
    return a

def is_invertible_matrix(matrix: np.ndarray, mod: int = ALPHABET_SIZE) -> bool:
    """
    Verifica se uma matriz é inversível mod 26.
    
    Args:
        matrix: Matriz a ser verificada
        mod: Módulo (padrão: 26)
        
    Returns:
        True se a matriz for inversível, False caso contrário
    """
    # Verificar se é matriz quadrada
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Calcular determinante
    det = int(round(np.linalg.det(matrix))) % mod
    
    # Verificar se o determinante é coprimo com o módulo
    return gcd(det, mod) == 1

def text_to_numbers(text: str) -> List[int]:
    """
    Converte texto para números (A=0, B=1, ..., Z=25).
    
    Args:
        text: Texto a ser convertido
        
    Returns:
        Lista de números correspondentes às letras
    """
    # Converter para maiúsculas e remover caracteres não alfabéticos
    text = re.sub(r'[^A-Za-z]', '', text.upper())
    return [LETTER_TO_NUM[letter] for letter in text]

def numbers_to_text(numbers: List[int]) -> str:
    """
    Converte números para texto.
    
    Args:
        numbers: Lista de números a serem convertidos
        
    Returns:
        Texto correspondente aos números
    """
    return ''.join(NUM_TO_LETTER[n % ALPHABET_SIZE] for n in numbers)

def encrypt_hill(plaintext: str, key_matrix: np.ndarray) -> str:
    """
    Encripta texto usando a cifra de Hill.
    
    Args:
        plaintext: Texto a ser encriptado
        key_matrix: Matriz chave
        
    Returns:
        Texto cifrado
    """
    # Converter texto para números
    numbers = text_to_numbers(plaintext)
    
    # Obter tamanho da matriz
    n = key_matrix.shape[0]
    
    # Adicionar padding se necessário
    if len(numbers) % n != 0:
        padding = n - (len(numbers) % n)
        numbers.extend([0] * padding)  # Adicionar 'A's como padding
    
    # Dividir em blocos de tamanho n
    blocks = [numbers[i:i+n] for i in range(0, len(numbers), n)]
    
    # Encriptar cada bloco
    encrypted_blocks = []
    for block in blocks:
        block_vector = np.array(block)
        encrypted_block = np.dot(key_matrix, block_vector) % ALPHABET_SIZE
        encrypted_blocks.extend(encrypted_block.tolist())
    
    # Converter números para texto
    return numbers_to_text(encrypted_blocks)

def decrypt_hill(ciphertext: str, key_matrix: np.ndarray) -> str:
    """
    Decripta texto usando a cifra de Hill.
    
    Args:
        ciphertext: Texto cifrado
        key_matrix: Matriz chave
        
    Returns:
        Texto decifrado
    """
    # Calcular inversa da matriz chave
    try:
        inverse_key = matrix_mod_inverse(key_matrix)
    except ValueError:
        raise ValueError("A matriz chave não é inversível mod 26")
    
    # Usar a função de encriptação com a matriz inversa
    return encrypt_hill(ciphertext, inverse_key)

# Implementação do ataque com texto conhecido
def known_plaintext_attack(plaintext: str, ciphertext: str, matrix_size: int) -> np.ndarray:
    """
    Implementa o ataque com texto claro conhecido.
    
    Args:
        plaintext: Texto claro conhecido
        ciphertext: Texto cifrado correspondente
        matrix_size: Tamanho da matriz (2, 3, 4 ou 5)
        
    Returns:
        Matriz chave candidata
        
    Raises:
        ValueError: Se os textos não tiverem tamanho suficiente ou a matriz não for inversível
    """
    # Converter textos para números
    p_nums = text_to_numbers(plaintext)
    c_nums = text_to_numbers(ciphertext)
    
    # Verificar se os textos têm tamanho suficiente
    if len(p_nums) < matrix_size * matrix_size or len(c_nums) < matrix_size * matrix_size:
        raise ValueError(f"Os textos devem ter pelo menos {matrix_size * matrix_size} caracteres")
    
    # Criar matrizes de texto claro e cifrado
    p_blocks = []
    c_blocks = []
    
    for i in range(0, matrix_size * matrix_size, matrix_size):
        p_block = p_nums[i:i+matrix_size]
        c_block = c_nums[i:i+matrix_size]
        p_blocks.append(p_block)
        c_blocks.append(c_block)
    
    P = np.array(p_blocks).T
    C = np.array(c_blocks).T
    
    # Calcular inversa da matriz de texto claro
    try:
        P_inv = matrix_mod_inverse(P)
    except ValueError:
        raise ValueError("A matriz de texto claro não é inversível mod 26")
    
    # Calcular matriz chave
    K = (C @ P_inv) % ALPHABET_SIZE
    
    return K
# Análise estatística básica de n-gramas
def ngram_frequency(text: str, n: int) -> Dict[str, float]:
    """
    Calcula a frequência de n-gramas em um texto.
    
    Args:
        text: Texto para análise
        n: Tamanho do n-grama
        
    Returns:
        Dicionário com n-gramas e suas frequências
    """
    # Converter para maiúsculas e remover caracteres não alfabéticos
    text = re.sub(r'[^A-Za-z]', '', text.upper())
    
    # Extrair n-gramas
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    
    # Contar ocorrências
    counter = Counter(ngrams)
    
    # Calcular frequências
    total = sum(counter.values())
    frequencies = {ngram: count / total for ngram, count in counter.items()}
    
    return frequencies

# Implementação da força bruta otimizada para matrizes 2x2 e 3x3
def generate_invertible_matrices(size: int, mod: int = ALPHABET_SIZE, limit: int = None) -> List[np.ndarray]:
    """
    Gera matrizes inversíveis de tamanho size x size mod 26.
    
    Args:
        size: Tamanho da matriz (2 ou 3)
        mod: Módulo (padrão: 26)
        limit: Limite de matrizes a gerar (opcional)
        
    Returns:
        Lista de matrizes inversíveis
        
    Raises:
        ValueError: Se o tamanho não for suportado
    """
    if size not in [2, 3]:
        raise ValueError("Apenas matrizes 2x2 e 3x3 são suportadas para força bruta")
    
    # Para matrizes 2x2, podemos gerar todas as combinações
    if size == 2:
        matrices = []
        for a in range(mod):
            for b in range(mod):
                for c in range(mod):
                    for d in range(mod):
                        matrix = np.array([[a, b], [c, d]])
                        if is_invertible_matrix(matrix, mod):
                            matrices.append(matrix)
        return matrices
    
    # Para matrizes 3x3, usar estratégias avançadas para reduzir o espaço de busca
    matrices = []
    
    # 1. Valores de determinante que são coprimos com 26
    valid_dets = [d for d in range(1, mod) if gcd(d, mod) == 1]
    
    # 2. Estratégia 1: Matrizes triangulares superiores com determinante válido
    for a in range(1, mod):
        if gcd(a, mod) != 1:  # a deve ser coprimo com 26
            continue
            
        for e in range(1, mod):
            if gcd(e, mod) != 1:  # e deve ser coprimo com 26
                continue
                
            for i in range(1, mod):
                if gcd(i, mod) != 1:  # i deve ser coprimo com 26
                    continue
                    
                # Verificar se o determinante é válido (a*e*i)
                det = (a * e * i) % mod
                if det not in valid_dets:
                    continue
                
                # Gerar elementos não-diagonais
                for b in range(0, mod, 3):  # Pular alguns valores para reduzir o espaço
                    for c in range(0, mod, 3):
                        for f in range(0, mod, 3):
                            for d in range(0, mod, 5):  # Elementos fora da diagonal principal
                                for g in range(0, mod, 5):
                                    for h in range(0, mod, 5):
                                        matrix = np.array([
                                            [a, b, c],
                                            [d, e, f],
                                            [g, h, i]
                                        ])
                                        
                                        # Verificar se é inversível (pode ser redundante, mas é uma verificação rápida)
                                        if is_invertible_matrix(matrix, mod):
                                            matrices.append(matrix)
                                            if limit and len(matrices) >= limit:
                                                return matrices
    
    # 3. Estratégia 2: Matrizes com estrutura de blocos
    # Usar matrizes 2x2 inversíveis como blocos
    small_matrices = generate_invertible_matrices(2, mod, 100)  # Limitar a 100 matrizes 2x2
    
    for small_matrix in small_matrices:
        # Criar matriz 3x3 com o bloco 2x2 no canto superior esquerdo
        for i in range(mod):
            for j in range(mod):
                for k in range(1, mod):
                    if gcd(k, mod) != 1:
                        continue
                        
                    matrix = np.zeros((3, 3), dtype=int)
                    matrix[0:2, 0:2] = small_matrix
                    matrix[0:2, 2] = [i, j]
                    matrix[2, 0:2] = [0, 0]  # Zeros para manter a estrutura de bloco
                    matrix[2, 2] = k
                    
                    if is_invertible_matrix(matrix, mod):
                        matrices.append(matrix)
                        if limit and len(matrices) >= limit:
                            return matrices
    
    # 4. Estratégia 3: Matrizes com padrões específicos
    patterns = [
        # Padrão diagonal dominante
        lambda a, b, c: np.array([
            [a, 1, 1],
            [1, b, 1],
            [1, 1, c]
        ]),
        # Padrão circulante
        lambda a, b, c: np.array([
            [a, b, c],
            [c, a, b],
            [b, c, a]
        ]),
        # Padrão simétrico
        lambda a, b, c: np.array([
            [a, b, c],
            [b, a, b],
            [c, b, a]
        ])
    ]
    
    for pattern_func in patterns:
        for a in range(1, mod):
            for b in range(mod):
                for c in range(mod):
                    matrix = pattern_func(a, b, c)
                    if is_invertible_matrix(matrix, mod):
                        matrices.append(matrix)
                        if limit and len(matrices) >= limit:
                            return matrices
    
    # 5. Estratégia 4: Usar análise de frequência para gerar matrizes mais prováveis
    # Baseado em frequências de letras em português
    freq_order = [0, 4, 14, 8, 18, 20, 17, 11, 3, 12, 15, 19, 21, 2, 5, 6, 7, 9, 10, 13, 16, 22, 23, 24, 25, 1]
    
    # Gerar matrizes com elementos mais frequentes nas posições mais importantes
    for i in range(5):  # Limitar a algumas combinações
        for j in range(5):
            for k in range(5):
                a = freq_order[i]
                e = freq_order[j]
                i_val = freq_order[k]
                
                if gcd(a, mod) != 1 or gcd(e, mod) != 1 or gcd(i_val, mod) != 1:
                    continue
                
                # Verificar se o determinante é válido
                det = (a * e * i_val) % mod
                if det not in valid_dets:
                    continue
                
                # Gerar elementos não-diagonais com valores mais prováveis
                for b_idx in range(3):
                    for c_idx in range(3):
                        for f_idx in range(3):
                            b = freq_order[b_idx]
                            c = freq_order[c_idx]
                            f = freq_order[f_idx]
                            
                            # Usar zeros para os outros elementos para simplificar
                            matrix = np.array([
                                [a, b, c],
                                [0, e, f],
                                [0, 0, i_val]
                            ])
                            
                            if is_invertible_matrix(matrix, mod):
                                matrices.append(matrix)
                                if limit and len(matrices) >= limit:
                                    return matrices
    
    # 6. Se ainda não tivermos matrizes suficientes, gerar algumas aleatoriamente
    # mas com viés para elementos mais prováveis
    while len(matrices) < (limit or 10000):
        # Escolher elementos diagonais com maior probabilidade de serem coprimos com 26
        a = np.random.choice([1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25])
        e = np.random.choice([1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25])
        i_val = np.random.choice([1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25])
        
        # Outros elementos podem ser quaisquer valores
        b, c, d, f, g, h = np.random.randint(0, mod, 6)
        
        matrix = np.array([
            [a, b, c],
            [d, e, f],
            [g, h, i_val]
        ])
        
        if is_invertible_matrix(matrix, mod):
            matrices.append(matrix)
            if limit and len(matrices) >= limit:
                return matrices
    
    return matrices

def brute_force_hill(ciphertext: str, matrix_size: int, language_model=None) -> List[Tuple[np.ndarray, str]]:
    """
    Implementa força bruta otimizada para quebrar a cifra de Hill.
    
    Args:
        ciphertext: Texto cifrado
        matrix_size: Tamanho da matriz (2 ou 3)
        language_model: Modelo de linguagem para validação (opcional)
        
    Returns:
        Lista de tuplas (matriz_chave, texto_decifrado) ordenadas por probabilidade
        
    Raises:
        ValueError: Se o tamanho da matriz não for suportado
    """
    if matrix_size not in [2, 3]:
        raise ValueError("Apenas matrizes 2x2 e 3x3 são suportadas para força bruta")
    
    # Gerar matrizes inversíveis
    matrices = generate_invertible_matrices(matrix_size)
    
    results = []
    for matrix in matrices:
        try:
            decrypted = decrypt_hill(ciphertext, matrix)
            # Se tiver modelo de linguagem, calcular probabilidade
            score = 0
            if language_model:
                score = language_model.score(decrypted)
            results.append((matrix, decrypted, score))
        except ValueError:
            continue
    
    # Ordenar por score (se disponível)
    if language_model:
        results.sort(key=lambda x: x[2], reverse=True)
    
    return [(matrix, decrypted) for matrix, decrypted, _ in results]
# Detecção de padrões no texto cifrado
def pattern_analysis(ciphertext: str, matrix_size: int) -> Dict[str, List[int]]:
    """
    Analisa padrões no texto cifrado para inferir informações sobre a chave.
    
    Args:
        ciphertext: Texto cifrado
        matrix_size: Tamanho da matriz
        
    Returns:
        Dicionário com padrões encontrados e suas posições
    """
    # Converter para maiúsculas e remover caracteres não alfabéticos
    text = re.sub(r'[^A-Za-z]', '', ciphertext.upper())
    
    patterns = {}
    
    # Procurar por repetições de n-gramas
    for n in range(2, matrix_size * 2 + 1):
        for i in range(len(text) - n + 1):
            ngram = text[i:i+n]
            if ngram not in patterns:
                patterns[ngram] = []
            patterns[ngram].append(i)
    
    # Filtrar apenas padrões que aparecem mais de uma vez
    patterns = {k: v for k, v in patterns.items() if len(v) > 1}
    
    # Calcular distâncias entre ocorrências
    for pattern, positions in patterns.items():
        distances = [positions[i+1] - positions[i] for i in range(len(positions) - 1)]
        patterns[pattern] = distances
    
    return patterns

# Validação de resultados usando dicionário português
class LanguageValidator:
    """Classe para validar textos usando um dicionário de palavras."""
    
    def __init__(self, dictionary_path: str = None):
        """
        Inicializa o validador de linguagem.
        
        Args:
            dictionary_path: Caminho para o arquivo de dicionário (opcional)
        """
        self.dictionary = set()
        if dictionary_path and os.path.exists(dictionary_path):
            try:
                with open(dictionary_path, 'r', encoding='utf-8') as f:
                    self.dictionary = set(word.strip().upper() for word in f)
            except Exception as e:
                print(f"Warning: Could not load dictionary from {dictionary_path}: {e}")
                # Create a small dictionary with common Portuguese words
                self.dictionary = {"DE", "A", "O", "QUE", "E", "DO", "DA", "EM", "UM", "PARA", "COM",
                                  "NAO", "UMA", "OS", "NO", "SE", "NA", "POR", "MAIS", "AS", "DOS"}
        else:
            # Create a small dictionary with common Portuguese words
            self.dictionary = {"DE", "A", "O", "QUE", "E", "DO", "DA", "EM", "UM", "PARA", "COM",
                              "NAO", "UMA", "OS", "NO", "SE", "NA", "POR", "MAIS", "AS", "DOS"}
        self.letter_freq = {
            'A': 0.1463, 'B': 0.0104, 'C': 0.0388, 'D': 0.0499,
            'E': 0.1257, 'F': 0.0102, 'G': 0.0130, 'H': 0.0128,
            'I': 0.0618, 'J': 0.0040, 'K': 0.0002, 'L': 0.0278,
            'M': 0.0474, 'N': 0.0505, 'O': 0.1073, 'P': 0.0252,
            'Q': 0.0120, 'R': 0.0653, 'S': 0.0781, 'T': 0.0434,
            'U': 0.0463, 'V': 0.0167, 'W': 0.0001, 'X': 0.0021,
            'Y': 0.0001, 'Z': 0.0047
        }
    
    def score_text(self, text: str) -> float:
        """
        Calcula um score para o texto baseado na frequência de letras.
        
        Args:
            text: Texto para avaliar
            
        Returns:
            Score do texto (maior é melhor)
        """
        # Converter para maiúsculas e remover caracteres não alfabéticos
        text = re.sub(r'[^A-Za-z]', '', text.upper())
        
        # Contar ocorrências de cada letra
        counter = Counter(text)
        
        # Calcular score baseado na frequência esperada
        score = 0
        for letter, count in counter.items():
            expected_freq = self.letter_freq.get(letter, 0)
            actual_freq = count / len(text)
            # Penalizar desvios da frequência esperada
            score -= abs(expected_freq - actual_freq)
        
        return score
    
    def validate_words(self, text: str) -> float:
        """
        Valida o texto verificando quantas palavras estão no dicionário.
        
        Args:
            text: Texto para validar
            
        Returns:
            Proporção de palavras válidas (0 a 1)
        """
        if not self.dictionary:
            return 0
        
        # Dividir em palavras
        words = re.findall(r'\b[A-Za-z]+\b', text.upper())
        
        if not words:
            return 0
        
        # Contar palavras válidas
        valid_words = sum(1 for word in words if word in self.dictionary)
        
        return valid_words / len(words)
# Classe principal para decifrar a Cifra de Hill
class HillCipherBreaker:
    """Classe principal para decifrar a Cifra de Hill."""
    
    def __init__(self, dictionary_path: str = None):
        """
        Inicializa o quebrador de cifra.
        
        Args:
            dictionary_path: Caminho para o arquivo de dicionário (opcional)
        """
        self.validator = LanguageValidator(dictionary_path)
    
    def break_cipher(self, ciphertext: str, matrix_size: int, known_plaintext: str = None) -> List[Tuple[np.ndarray, str, float]]:
        """
        Tenta quebrar a cifra usando vários métodos.
        
        Args:
            ciphertext: Texto cifrado
            matrix_size: Tamanho da matriz
            known_plaintext: Texto claro conhecido (opcional)
            
        Returns:
            Lista de tuplas (matriz_chave, texto_decifrado, score) ordenadas por score
        """
        results = []
        
        # 1. Se tiver texto conhecido, usar ataque com texto conhecido
        if known_plaintext:
            try:
                key_matrix = known_plaintext_attack(known_plaintext, ciphertext, matrix_size)
                decrypted = decrypt_hill(ciphertext, key_matrix)
                score = self.validator.score_text(decrypted)
                results.append((key_matrix, decrypted, score))
            except ValueError as e:
                print(f"Erro no ataque com texto conhecido: {e}")
        
        # 2. Para matrizes pequenas, usar força bruta
        if matrix_size in [2, 3]:
            try:
                brute_force_results = brute_force_hill(ciphertext, matrix_size)
                for matrix, decrypted in brute_force_results:
                    score = self.validator.score_text(decrypted)
                    results.append((matrix, decrypted, score))
            except ValueError as e:
                print(f"Erro na força bruta: {e}")
        
        # 3. Análise de padrões
        patterns = pattern_analysis(ciphertext, matrix_size)
        # Usar informações de padrões para refinar resultados
        # (Implementação simplificada)
        
        # Ordenar resultados por score
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results
    
    def generate_report(self, results: List[Tuple[np.ndarray, str, float]], ciphertext: str, matrix_size: int) -> str:
        """
        Gera um relatório dos resultados.
        
        Args:
            results: Lista de resultados (matriz_chave, texto_decifrado, score)
            ciphertext: Texto cifrado original
            matrix_size: Tamanho da matriz
            
        Returns:
            Relatório formatado
        """
        report = []
        report.append("=== RELATÓRIO DE DECIFRAGEM DA CIFRA DE HILL ===")
        report.append(f"Tamanho da matriz: {matrix_size}x{matrix_size}")
        report.append(f"Texto cifrado: {ciphertext[:50]}..." if len(ciphertext) > 50 else ciphertext)
        report.append(f"Número de resultados: {len(results)}")
        
        if results:
            report.append("\nMelhores resultados:")
            for i, (matrix, decrypted, score) in enumerate(results[:5], 1):
                report.append(f"\n--- Resultado #{i} (Score: {score:.4f}) ---")
                report.append(f"Matriz chave:\n{matrix}")
                report.append(f"Texto decifrado: {decrypted[:100]}..." if len(decrypted) > 100 else decrypted)
        else:
            report.append("\nNenhum resultado encontrado.")
        
        return "\n".join(report)
# Função principal
def main():
    """Função principal do programa."""
    print("=== Hill Cipher Breaker ===")
    
    # Diretórios de textos
    known_dir = "../textos_conhecidos"
    unknown_dir = "../textos_desconhecidos"
    
    # Verificar se os diretórios existem
    if not os.path.exists(known_dir) or not os.path.exists(unknown_dir):
        print(f"Erro: Diretórios {known_dir} ou {unknown_dir} não encontrados.")
        return
    
    # Criar instância do quebrador de cifra
    breaker = HillCipherBreaker()
    
    # Processar textos conhecidos
    print("\nProcessando textos conhecidos...")
    for size in [2, 3, 4, 5]:
        cipher_path = os.path.join(known_dir, "Cifrado", "Hill", f"Grupo02_{size}_texto_cifrado.txt")
        
        if os.path.exists(cipher_path):
            with open(cipher_path, 'r') as f:
                ciphertext = f.read().strip()
            
            print(f"\nQuebrando cifra {size}x{size}...")
            results = breaker.break_cipher(ciphertext, size)
            
            if results:
                report = breaker.generate_report(results, ciphertext, size)
                print(report)
                
                # Salvar relatório
                report_path = f"relatorios/basico/conhecidos/hill_{size}x{size}/relatorio.txt"
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                with open(report_path, 'w') as f:
                    f.write(report)
                print(f"Relatório salvo em {report_path}")
    
    # Processar textos desconhecidos
    print("\nProcessando textos desconhecidos...")
    for size in [2, 3, 4, 5]:
        cipher_path = os.path.join(unknown_dir, "Cifrado", "Hill", f"Grupo02_{size}_texto_cifrado.txt")
        
        if os.path.exists(cipher_path):
            with open(cipher_path, 'r') as f:
                ciphertext = f.read().strip()
            
            print(f"\nQuebrando cifra {size}x{size}...")
            results = breaker.break_cipher(ciphertext, size)
            
            if results:
                report = breaker.generate_report(results, ciphertext, size)
                print(report)
                
                # Salvar relatório
                report_path = f"relatorios/basico/desconhecidos/hill_{size}x{size}/relatorio.txt"
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                with open(report_path, 'w') as f:
                    f.write(report)
                print(f"Relatório salvo em {report_path}")

if __name__ == "__main__":
    main()

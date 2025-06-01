#!/usr/bin/env python3
"""
Hill Cipher Breaker (Improved) - Programa para decifrar a Cifra de Hill

Este programa implementa diversos métodos para quebrar a cifra de Hill,
incluindo ataque com texto conhecido, análise estatística, força bruta
e detecção de padrões, com suporte aprimorado para matrizes 4x4 e 5x5.

Autor: Amazon Q
"""

import numpy as np
import os
import re
import math
import itertools
from collections import Counter
import time
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional, Union, Set

# Constantes
ALPHABET_SIZE = 26
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
LETTER_TO_NUM = {letter: idx for idx, letter in enumerate(ALPHABET)}
NUM_TO_LETTER = {idx: letter for idx, letter in enumerate(ALPHABET)}

# Frequências de letras em português
PORTUGUESE_LETTER_FREQ = {
    'A': 0.1463, 'B': 0.0104, 'C': 0.0388, 'D': 0.0499,
    'E': 0.1257, 'F': 0.0102, 'G': 0.0130, 'H': 0.0128,
    'I': 0.0618, 'J': 0.0040, 'K': 0.0002, 'L': 0.0278,
    'M': 0.0474, 'N': 0.0505, 'O': 0.1073, 'P': 0.0252,
    'Q': 0.0120, 'R': 0.0653, 'S': 0.0781, 'T': 0.0434,
    'U': 0.0463, 'V': 0.0167, 'W': 0.0001, 'X': 0.0021,
    'Y': 0.0001, 'Z': 0.0047
}

# Bigramas comuns em português
COMMON_BIGRAMS = [
    'DE', 'RA', 'ES', 'OS', 'AR', 'QU', 'NT', 'EN', 'ER', 'TE', 
    'CO', 'RE', 'AS', 'TA', 'DO', 'OR', 'ME', 'MA', 'EM', 'ND'
]

# Trigramas comuns em português
COMMON_TRIGRAMS = [
    'QUE', 'EST', 'NTE', 'COM', 'ENT', 'ARA', 'CON', 'TEM', 'ADE', 'RES',
    'DOS', 'NTO', 'PAR', 'POR', 'TER', 'ERA', 'MEN', 'UMA', 'STA', 'DES'
]

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
# Análise estatística de n-gramas
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

def score_ngrams(text: str, common_ngrams: List[str], n: int) -> float:
    """
    Calcula um score baseado na presença de n-gramas comuns.
    
    Args:
        text: Texto para avaliar
        common_ngrams: Lista de n-gramas comuns
        n: Tamanho do n-grama
        
    Returns:
        Score do texto (maior é melhor)
    """
    # Converter para maiúsculas e remover caracteres não alfabéticos
    text = re.sub(r'[^A-Za-z]', '', text.upper())
    
    # Extrair n-gramas
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    
    # Contar ocorrências de n-gramas comuns
    score = sum(1 for ngram in ngrams if ngram in common_ngrams)
    
    # Normalizar pelo tamanho do texto
    return score / (len(text) - n + 1)

# Implementação da força bruta otimizada para matrizes 2x2 e 3x3
def generate_invertible_matrices(size: int, mod: int = ALPHABET_SIZE) -> List[np.ndarray]:
    """
    Gera matrizes inversíveis de tamanho size x size mod 26.
    
    Args:
        size: Tamanho da matriz (2 ou 3)
        mod: Módulo (padrão: 26)
        
    Returns:
        Lista de matrizes inversíveis
        
    Raises:
        ValueError: Se o tamanho não for suportado
    """
    if size not in [2, 3]:
        raise ValueError("Apenas matrizes 2x2 e 3x3 são suportadas para geração completa")
    
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
    
    # Para matrizes 3x3, usar uma abordagem mais seletiva
    matrices = []
    # Gerar apenas matrizes com determinantes coprimos com 26
    valid_dets = [d for d in range(1, mod) if gcd(d, mod) == 1]
    
    # Gerar algumas matrizes para cada determinante válido
    for det_val in valid_dets:
        count = 0
        max_per_det = 100  # Limitar o número de matrizes por determinante
        
        # Gerar matrizes aleatórias com o determinante desejado
        while count < max_per_det:
            matrix = np.random.randint(0, mod, (size, size))
            if is_invertible_matrix(matrix, mod):
                matrices.append(matrix)
                count += 1
    
    return matrices

def brute_force_hill(ciphertext: str, matrix_size: int, language_validator=None) -> List[Tuple[np.ndarray, str, float]]:
    """
    Implementa força bruta otimizada para quebrar a cifra de Hill.
    
    Args:
        ciphertext: Texto cifrado
        matrix_size: Tamanho da matriz (2 ou 3)
        language_validator: Validador de linguagem para pontuação (opcional)
        
    Returns:
        Lista de tuplas (matriz_chave, texto_decifrado, score) ordenadas por score
        
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
            
            # Calcular score
            score = 0
            if language_validator:
                score = language_validator.score_text(decrypted)
            
            results.append((matrix, decrypted, score))
        except ValueError:
            continue
    
    # Ordenar por score
    results.sort(key=lambda x: x[2], reverse=True)
    
    return results
# Detecção de padrões no texto cifrado
def pattern_analysis(ciphertext: str, matrix_size: int) -> Dict[str, List[int]]:
    """
    Analisa padrões no texto cifrado para inferir informações sobre a chave.
    
    Args:
        ciphertext: Texto cifrado
        matrix_size: Tamanho da matriz
        
    Returns:
        Dicionário com padrões encontrados e suas distâncias
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
    pattern_distances = {}
    for pattern, positions in patterns.items():
        distances = [positions[i+1] - positions[i] for i in range(len(positions) - 1)]
        pattern_distances[pattern] = distances
    
    return pattern_distances

def analyze_pattern_distances(pattern_distances: Dict[str, List[int]], matrix_size: int) -> Dict[str, float]:
    """
    Analisa as distâncias entre padrões para inferir informações sobre a chave.
    
    Args:
        pattern_distances: Dicionário com padrões e suas distâncias
        matrix_size: Tamanho da matriz
        
    Returns:
        Dicionário com padrões e suas pontuações
    """
    pattern_scores = {}
    
    for pattern, distances in pattern_distances.items():
        # Verificar se as distâncias são múltiplos do tamanho da matriz
        multiple_count = sum(1 for d in distances if d % matrix_size == 0)
        
        # Calcular proporção de distâncias que são múltiplos do tamanho da matriz
        if distances:
            score = multiple_count / len(distances)
            pattern_scores[pattern] = score
    
    return pattern_scores

# Técnicas de redução do espaço de busca para matrizes maiores
def generate_promising_matrices(matrix_size: int, mod: int = ALPHABET_SIZE, limit: int = 1000) -> List[np.ndarray]:
    """
    Gera matrizes promissoras para matrizes maiores (4x4 e 5x5).
    
    Args:
        matrix_size: Tamanho da matriz (4 ou 5)
        mod: Módulo (padrão: 26)
        limit: Limite de matrizes a gerar
        
    Returns:
        Lista de matrizes promissoras
    """
    if matrix_size not in [4, 5]:
        raise ValueError("Esta função é para matrizes 4x4 e 5x5")
    
    matrices = []
    count = 0
    
    # Gerar matrizes com propriedades desejáveis
    while count < limit:
        # Estratégia 1: Usar blocos de matrizes 2x2 conhecidas
        if matrix_size == 4:
            # Criar matriz 4x4 a partir de blocos 2x2
            block1 = np.random.randint(0, mod, (2, 2))
            block2 = np.random.randint(0, mod, (2, 2))
            block3 = np.random.randint(0, mod, (2, 2))
            block4 = np.random.randint(0, mod, (2, 2))
            
            matrix = np.block([[block1, block2], [block3, block4]])
        else:  # 5x5
            # Para 5x5, usar uma abordagem diferente
            matrix = np.random.randint(0, mod, (matrix_size, matrix_size))
            
            # Garantir que alguns elementos sejam zero para reduzir complexidade
            zero_positions = np.random.choice(matrix_size*matrix_size, size=matrix_size, replace=False)
            for pos in zero_positions:
                i, j = pos // matrix_size, pos % matrix_size
                matrix[i, j] = 0
        
        # Verificar se é inversível
        if is_invertible_matrix(matrix, mod):
            matrices.append(matrix)
            count += 1
    
    return matrices

# Estratégias de pontuação (Scoring) para matrizes maiores
class ScoringStrategy:
    """Classe para implementar estratégias de pontuação para candidatos."""
    
    def __init__(self, language_validator=None):
        """
        Inicializa a estratégia de pontuação.
        
        Args:
            language_validator: Validador de linguagem (opcional)
        """
        self.language_validator = language_validator
    
    def score_candidate(self, decrypted_text: str) -> float:
        """
        Calcula um score para o texto decifrado.
        
        Args:
            decrypted_text: Texto decifrado
            
        Returns:
            Score do texto (maior é melhor)
        """
        score = 0
        
        # 1. Pontuação baseada em frequência de letras
        if self.language_validator:
            score += self.language_validator.score_text(decrypted_text)
        
        # 2. Pontuação baseada em n-gramas comuns
        bigram_score = score_ngrams(decrypted_text, COMMON_BIGRAMS, 2)
        trigram_score = score_ngrams(decrypted_text, COMMON_TRIGRAMS, 3)
        
        # Pesos para diferentes componentes do score
        score += 5 * bigram_score + 10 * trigram_score
        
        # 3. Penalizar sequências improváveis
        unlikely_patterns = ['ZZ', 'QQ', 'JJ', 'XX', 'WW']
        for pattern in unlikely_patterns:
            score -= 0.5 * decrypted_text.count(pattern)
        
        return score

# Heurística de Janela (Shutter) para matrizes maiores
class ShutterHeuristic:
    """Implementa a heurística de janela para focar em regiões promissoras."""
    
    def __init__(self, matrix_size: int, window_size: int = 10):
        """
        Inicializa a heurística de janela.
        
        Args:
            matrix_size: Tamanho da matriz
            window_size: Tamanho da janela
        """
        self.matrix_size = matrix_size
        self.window_size = window_size
        self.promising_regions = []
        self.history = []
    
    def update_regions(self, results: List[Tuple[np.ndarray, str, float]]):
        """
        Atualiza regiões promissoras com base nos resultados.
        
        Args:
            results: Lista de resultados (matriz, texto, score)
        """
        # Ordenar resultados por score
        sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
        
        # Adicionar melhores resultados ao histórico
        self.history.extend(sorted_results[:self.window_size])
        
        # Limitar tamanho do histórico
        if len(self.history) > 100:
            self.history = sorted(self.history, key=lambda x: x[2], reverse=True)[:50]
        
        # Identificar regiões promissoras
        self.promising_regions = [matrix for matrix, _, _ in self.history[:self.window_size]]
    
    def generate_candidates(self, num_candidates: int = 20) -> List[np.ndarray]:
        """
        Gera candidatos baseados em regiões promissoras.
        
        Args:
            num_candidates: Número de candidatos a gerar
            
        Returns:
            Lista de matrizes candidatas
        """
        candidates = []
        
        # Se não tiver regiões promissoras, gerar aleatoriamente
        if not self.promising_regions:
            return generate_promising_matrices(self.matrix_size, limit=num_candidates)
        
        # Gerar variações das regiões promissoras
        for base_matrix in self.promising_regions[:5]:  # Usar as 5 melhores regiões
            for _ in range(num_candidates // 5):
                # Criar uma variação da matriz base
                variation = base_matrix.copy()
                
                # Modificar alguns elementos aleatoriamente
                num_changes = np.random.randint(1, 4)  # 1 a 3 mudanças
                for _ in range(num_changes):
                    i, j = np.random.randint(0, self.matrix_size, 2)
                    variation[i, j] = (variation[i, j] + np.random.randint(1, 5)) % ALPHABET_SIZE
                
                # Verificar se é inversível
                if is_invertible_matrix(variation):
                    candidates.append(variation)
        
        return candidates
# Validação de resultados usando dicionário português
class LanguageValidator:
    """Classe para validar textos usando um dicionário de palavras e estatísticas linguísticas."""
    
    def __init__(self, dictionary_path: str = None):
        """
        Inicializa o validador de linguagem.
        
        Args:
            dictionary_path: Caminho para o arquivo de dicionário (opcional)
        """
        self.dictionary = set()
        if dictionary_path and os.path.exists(dictionary_path):
            with open(dictionary_path, 'r', encoding='utf-8') as f:
                self.dictionary = set(word.strip().upper() for word in f)
        
        # Frequências de letras em português
        self.letter_freq = PORTUGUESE_LETTER_FREQ
    
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
        
        if not text:
            return -float('inf')
        
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
    
    def comprehensive_score(self, text: str) -> float:
        """
        Calcula um score abrangente para o texto.
        
        Args:
            text: Texto para avaliar
            
        Returns:
            Score abrangente (maior é melhor)
        """
        # Score baseado em frequência de letras
        letter_score = self.score_text(text)
        
        # Score baseado em n-gramas
        bigram_score = score_ngrams(text, COMMON_BIGRAMS, 2)
        trigram_score = score_ngrams(text, COMMON_TRIGRAMS, 3)
        
        # Score baseado em palavras válidas
        word_score = self.validate_words(text)
        
        # Combinar scores com pesos
        return letter_score + 5 * bigram_score + 10 * trigram_score + 20 * word_score

# Implementação de técnicas avançadas para matrizes 4x4 e 5x5
class AdvancedHillBreaker:
    """Implementa técnicas avançadas para quebrar cifras Hill com matrizes grandes."""
    
    def __init__(self, matrix_size: int, language_validator: LanguageValidator):
        """
        Inicializa o quebrador avançado.
        
        Args:
            matrix_size: Tamanho da matriz (4 ou 5)
            language_validator: Validador de linguagem
        """
        self.matrix_size = matrix_size
        self.language_validator = language_validator
        self.scoring_strategy = ScoringStrategy(language_validator)
        self.shutter_heuristic = ShutterHeuristic(matrix_size)
        self.best_candidates = []
        self.iteration = 0
        self.max_iterations = 50
    
    def break_cipher(self, ciphertext: str, known_plaintext: str = None) -> List[Tuple[np.ndarray, str, float]]:
        """
        Tenta quebrar a cifra usando técnicas avançadas.
        
        Args:
            ciphertext: Texto cifrado
            known_plaintext: Texto claro conhecido (opcional)
            
        Returns:
            Lista de tuplas (matriz_chave, texto_decifrado, score) ordenadas por score
        """
        # Se tiver texto conhecido, usar ataque com texto conhecido
        if known_plaintext:
            try:
                key_matrix = known_plaintext_attack(known_plaintext, ciphertext, self.matrix_size)
                decrypted = decrypt_hill(ciphertext, key_matrix)
                score = self.language_validator.comprehensive_score(decrypted)
                return [(key_matrix, decrypted, score)]
            except ValueError:
                print("Ataque com texto conhecido falhou, tentando outras abordagens...")
        
        # Analisar padrões no texto cifrado
        pattern_distances = pattern_analysis(ciphertext, self.matrix_size)
        pattern_scores = analyze_pattern_distances(pattern_distances, self.matrix_size)
        
        # Inicializar com algumas matrizes promissoras
        candidates = generate_promising_matrices(self.matrix_size, limit=50)
        
        # Processo iterativo de refinamento
        results = []
        for self.iteration in range(self.max_iterations):
            print(f"Iteração {self.iteration + 1}/{self.max_iterations}")
            
            # Avaliar candidatos atuais
            iteration_results = []
            for matrix in candidates:
                try:
                    decrypted = decrypt_hill(ciphertext, matrix)
                    score = self.scoring_strategy.score_candidate(decrypted)
                    iteration_results.append((matrix, decrypted, score))
                except ValueError:
                    continue
            
            # Atualizar melhores resultados
            results.extend(iteration_results)
            results.sort(key=lambda x: x[2], reverse=True)
            results = results[:100]  # Manter apenas os 100 melhores
            
            # Atualizar heurística de janela
            self.shutter_heuristic.update_regions(iteration_results)
            
            # Gerar novos candidatos baseados nas regiões promissoras
            candidates = self.shutter_heuristic.generate_candidates(num_candidates=50)
            
            # Verificar critério de parada
            if self.iteration > 10 and results and results[0][2] > 0:
                # Se temos um bom candidato após 10 iterações, podemos parar
                break
        
        return results

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
                score = self.validator.comprehensive_score(decrypted)
                results.append((key_matrix, decrypted, score))
            except ValueError as e:
                print(f"Erro no ataque com texto conhecido: {e}")
        
        # 2. Para matrizes pequenas, usar força bruta
        if matrix_size in [2, 3]:
            try:
                brute_force_results = brute_force_hill(ciphertext, matrix_size, self.validator)
                results.extend(brute_force_results)
            except ValueError as e:
                print(f"Erro na força bruta: {e}")
        
        # 3. Para matrizes maiores, usar técnicas avançadas
        elif matrix_size in [4, 5]:
            try:
                advanced_breaker = AdvancedHillBreaker(matrix_size, self.validator)
                advanced_results = advanced_breaker.break_cipher(ciphertext, known_plaintext)
                results.extend(advanced_results)
            except Exception as e:
                print(f"Erro nas técnicas avançadas: {e}")
        
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
    print("=== Hill Cipher Breaker (Improved) ===")
    
    # Diretórios de textos
    known_dir = "textos_conhecidos"
    unknown_dir = "textos_desconhecidos"
    
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
            start_time = time.time()
            results = breaker.break_cipher(ciphertext, size)
            elapsed_time = time.time() - start_time
            
            if results:
                report = breaker.generate_report(results, ciphertext, size)
                print(report)
                print(f"Tempo de execução: {elapsed_time:.2f} segundos")
                
                # Salvar relatório
                report_path = f"relatorio_improved_hill_{size}x{size}.txt"
                with open(report_path, 'w') as f:
                    f.write(report)
                    f.write(f"\n\nTempo de execução: {elapsed_time:.2f} segundos")
                print(f"Relatório salvo em {report_path}")
    
    # Processar textos desconhecidos
    print("\nProcessando textos desconhecidos...")
    for size in [2, 3, 4, 5]:
        cipher_path = os.path.join(unknown_dir, "Cifrado", "Hill", f"Grupo02_{size}_texto_cifrado.txt")
        
        if os.path.exists(cipher_path):
            with open(cipher_path, 'r') as f:
                ciphertext = f.read().strip()
            
            print(f"\nQuebrando cifra {size}x{size}...")
            start_time = time.time()
            results = breaker.break_cipher(ciphertext, size)
            elapsed_time = time.time() - start_time
            
            if results:
                report = breaker.generate_report(results, ciphertext, size)
                print(report)
                print(f"Tempo de execução: {elapsed_time:.2f} segundos")
                
                # Salvar relatório
                report_path = f"relatorio_improved_desconhecido_hill_{size}x{size}.txt"
                with open(report_path, 'w') as f:
                    f.write(report)
                    f.write(f"\n\nTempo de execução: {elapsed_time:.2f} segundos")
                print(f"Relatório salvo em {report_path}")

if __name__ == "__main__":
    main()

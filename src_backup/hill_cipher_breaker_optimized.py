#!/usr/bin/env python3
"""
Hill Cipher Breaker (Optimized) - Programa otimizado para decifrar a Cifra de Hill

Este programa implementa métodos avançados e otimizados para quebrar a cifra de Hill,
com melhorias na análise de palavras, ataque com texto conhecido, sistema de pontuação
e pós-processamento do texto decifrado.

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
import concurrent.futures
import pickle
import urllib.request
import ssl
import logging
from typing import List, Dict, Tuple, Optional, Union, Set
import urllib.request
import pickle
import random
import ssl
import concurrent.futures

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
    'CO', 'RE', 'AS', 'TA', 'DO', 'OR', 'ME', 'MA', 'EM', 'ND',
    'SE', 'NO', 'UM', 'PA', 'EL', 'AM', 'AN', 'TO', 'CA', 'AL'
]

# Trigramas comuns em português
COMMON_TRIGRAMS = [
    'QUE', 'EST', 'NTE', 'COM', 'ENT', 'ARA', 'CON', 'TEM', 'ADE', 'RES',
    'DOS', 'NTO', 'PAR', 'POR', 'TER', 'ERA', 'MEN', 'UMA', 'STA', 'DES',
    'AND', 'ADA', 'NTA', 'NHA', 'AVA', 'ADO', 'ORA', 'IDA', 'AIS', 'NTO'
]

# Quadrigramas comuns em português
COMMON_QUADGRAMS = [
    'MENT', 'ENTE', 'PARA', 'ANDO', 'ANDO', 'ESTA', 'AVEL', 'IDAD', 'NTOS',
    'ENTE', 'ANDO', 'ENTE', 'MENT', 'PARA', 'ESTA', 'AVEL', 'IDAD', 'NTOS',
    'ENTE', 'ANDO', 'ENTE', 'MENT', 'PARA', 'ESTA', 'AVEL', 'IDAD', 'NTOS'
]

# Palavras comuns em português (para segmentação de texto)
COMMON_WORDS = [
    'DE', 'A', 'O', 'QUE', 'E', 'DO', 'DA', 'EM', 'UM', 'PARA',
    'COM', 'NAO', 'UMA', 'OS', 'NO', 'SE', 'NA', 'POR', 'MAIS', 'AS',
    'DOS', 'COMO', 'MAS', 'AO', 'ELE', 'DAS', 'SEU', 'SUA', 'OU', 'TER',
    'QUANDO', 'MUITO', 'NOS', 'JA', 'EU', 'TAMBEM', 'SO', 'PELO', 'PELA',
    'ATE', 'ISSO', 'ELA', 'ENTRE', 'DEPOIS', 'SEM', 'MESMO', 'AOS', 'SEUS',
    'QUEM', 'NAS', 'ME', 'ESSE', 'ELES', 'VOCE', 'ESSA', 'NUM', 'NEM', 'SUAS'
]

# URL para baixar dicionário de português
DICT_URL = "https://www.ime.usp.br/~pf/dicios/br-sem-acentos.txt"
DICT_PATH = "portuguese_dict.txt"

# Cache para matrizes inversíveis
INVERTIBLE_MATRICES_CACHE = {}

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
    # Otimização: usar algoritmo estendido de Euclides
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        else:
            gcd, x, y = extended_gcd(b % a, a)
            return gcd, y - (b // a) * x, x
    
    gcd, x, y = extended_gcd(a % m, m)
    if gcd != 1:
        raise ValueError(f"O inverso modular de {a} mod {m} não existe")
    else:
        return x % m

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
def known_plaintext_attack(plaintext: str, ciphertext: str, matrix_size: int) -> List[np.ndarray]:
    """
    Implementa o ataque com texto claro conhecido, tentando múltiplas combinações.
    
    Args:
        plaintext: Texto claro conhecido
        ciphertext: Texto cifrado correspondente
        matrix_size: Tamanho da matriz (2, 3, 4 ou 5)
        
    Returns:
        Lista de matrizes chave candidatas
        
    Raises:
        ValueError: Se os textos não tiverem tamanho suficiente
    """
    # Converter textos para números
    p_nums = text_to_numbers(plaintext)
    c_nums = text_to_numbers(ciphertext)
    
    # Verificar se os textos têm tamanho suficiente
    if len(p_nums) < matrix_size * matrix_size or len(c_nums) < matrix_size * matrix_size:
        raise ValueError(f"Os textos devem ter pelo menos {matrix_size * matrix_size} caracteres")
    
    # Tentar diferentes posições iniciais
    candidates = []
    max_start = min(len(p_nums), len(c_nums)) - matrix_size * matrix_size
    
    for start_pos in range(0, max_start + 1, matrix_size):
        # Criar matrizes de texto claro e cifrado
        p_blocks = []
        c_blocks = []
        
        for i in range(start_pos, start_pos + matrix_size * matrix_size, matrix_size):
            if i + matrix_size <= len(p_nums) and i + matrix_size <= len(c_nums):
                p_block = p_nums[i:i+matrix_size]
                c_block = c_nums[i:i+matrix_size]
                p_blocks.append(p_block)
                c_blocks.append(c_block)
        
        if len(p_blocks) == matrix_size:
            P = np.array(p_blocks).T
            C = np.array(c_blocks).T
            
            # Verificar se a matriz de texto claro é inversível
            if is_invertible_matrix(P):
                try:
                    # Calcular inversa da matriz de texto claro
                    P_inv = matrix_mod_inverse(P)
                    
                    # Calcular matriz chave
                    K = (C @ P_inv) % ALPHABET_SIZE
                    
                    # Verificar se a matriz chave é inversível
                    if is_invertible_matrix(K):
                        candidates.append(K)
                except ValueError:
                    continue
    
    return candidates

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
    
    if not ngrams:
        return 0
    
    # Contar ocorrências de n-gramas comuns
    score = sum(1 for ngram in ngrams if ngram in common_ngrams)
    
    # Normalizar pelo tamanho do texto
    return score / len(ngrams)

# Segmentação de texto sem espaços
def segment_text(text: str, max_words: int = 100) -> str:
    """
    Segmenta um texto sem espaços em palavras.
    
    Args:
        text: Texto sem espaços
        max_words: Número máximo de palavras a identificar
        
    Returns:
        Texto com espaços entre palavras
    """
    # Converter para maiúsculas
    text = text.upper()
    
    # Inicializar resultado
    result = []
    
    # Posição atual no texto
    pos = 0
    
    # Contador de palavras
    word_count = 0
    
    while pos < len(text) and word_count < max_words:
        # Tentar encontrar a palavra mais longa possível
        found = False
        
        # Tentar palavras de tamanho 2 a 10
        for length in range(10, 1, -1):
            if pos + length <= len(text):
                word = text[pos:pos+length]
                if word in COMMON_WORDS:
                    result.append(word)
                    pos += length
                    found = True
                    word_count += 1
                    break
        
        # Se não encontrou palavra, avançar um caractere
        if not found:
            result.append(text[pos])
            pos += 1
    
    # Adicionar o resto do texto
    if pos < len(text):
        result.append(text[pos:])
    
    # Juntar resultado com espaços
    return ' '.join(result)

# Classe para gerenciar o dicionário de palavras em português
class PortugueseDictionary:
    """Classe para gerenciar o dicionário de palavras em português."""
    
    def __init__(self, dict_path: str = DICT_PATH):
        """
        Inicializa o dicionário.
        
        Args:
            dict_path: Caminho para o arquivo de dicionário
        """
        self.dict_path = dict_path
        self.words = set()
        self.word_prefixes = set()  # Prefixos de palavras para segmentação
        self.word_suffixes = set()  # Sufixos de palavras para segmentação
        self.common_word_pairs = set()  # Pares de palavras comuns
        self.load_dictionary()
    
    def load_dictionary(self):
        """Carrega o dicionário de palavras."""
        # Verificar se o arquivo existe
        if not os.path.exists(self.dict_path):
            try:
                print(f"Baixando dicionário de português de {DICT_URL}...")
                # Ignorar verificação de certificado SSL para contornar o erro
                ssl._create_default_https_context = ssl._create_unverified_context
                urllib.request.urlretrieve(DICT_URL, self.dict_path)
                print("Download concluído.")
            except Exception as e:
                print(f"Erro ao baixar dicionário: {e}")
                # Criar um dicionário mínimo com palavras comuns
                self.create_minimal_dictionary()
                return
        
        # Carregar palavras do arquivo
        try:
            with open(self.dict_path, 'r', encoding='latin-1') as f:
                self.words = set(word.strip().upper() for word in f if word.strip())
            
            # Criar conjunto de prefixos e sufixos para todas as palavras
            for word in self.words:
                # Adicionar prefixos (começo da palavra)
                for i in range(1, len(word) + 1):
                    self.word_prefixes.add(word[:i])
                
                # Adicionar sufixos (final da palavra)
                for i in range(1, len(word) + 1):
                    self.word_suffixes.add(word[-i:])
            
            # Criar pares de palavras comuns
            for word1 in COMMON_WORDS:
                for word2 in COMMON_WORDS:
                    if len(word1) > 2 and len(word2) > 2:  # Apenas palavras com mais de 2 letras
                        self.common_word_pairs.add(word1 + word2)
            
            print(f"Dicionário carregado com {len(self.words)} palavras.")
        except Exception as e:
            print(f"Erro ao carregar dicionário: {e}")
            self.create_minimal_dictionary()
    
    def create_minimal_dictionary(self):
        """Cria um dicionário mínimo com palavras comuns em português."""
        common_words = [
            "A", "DE", "QUE", "O", "E", "DO", "DA", "EM", "UM", "PARA",
            "COM", "NAO", "UMA", "OS", "NO", "SE", "NA", "POR", "MAIS", "AS",
            "DOS", "COMO", "MAS", "AO", "ELE", "DAS", "A", "SEU", "SUA", "OU",
            "QUANDO", "MUITO", "NOS", "JA", "EU", "TAMBEM", "SO", "PELO", "PELA",
            "ATE", "ISSO", "ELA", "ENTRE", "DEPOIS", "SEM", "MESMO", "AOS", "SEUS",
            "QUEM", "NAS", "ME", "ESSE", "ELES", "VOCE", "ESSA", "NUM", "NEM", "SUAS",
            "MEU", "AS", "MINHA", "NUMA", "PELOS", "ELAS", "QUAL", "NESTE", "PELAS",
            "ESTE", "FOSSE", "DELE", "TU", "TE", "VOCES", "VOS", "LHE", "LHES", "MEUS",
            "MINHAS", "TEU", "TUA", "TEUS", "TUAS", "NOSSO", "NOSSA", "NOSSOS", "NOSSAS",
            "DELA", "DELES", "DELAS", "ESTA", "ESTES", "ESTAS", "AQUELE", "AQUELA",
            "AQUELES", "AQUELAS", "ISTO", "AQUILO", "ESTOU", "ESTA", "ESTAMOS", "ESTAO",
            "ESTIVE", "ESTEVE", "ESTIVEMOS", "ESTIVERAM", "ESTAVA", "ESTAVAMOS", "ESTAVAM",
            "ESTIVERA", "ESTIVERAMOS", "ESTEJA", "ESTEJAMOS", "ESTEJAM", "ESTIVESSE",
            "ESTIVESSEMOS", "ESTIVESSEM", "ESTIVER", "ESTIVERMOS", "ESTIVEREM"
        ]
        self.words = set(common_words)
        
        # Criar conjunto de prefixos e sufixos para todas as palavras
        for word in self.words:
            # Adicionar prefixos (começo da palavra)
            for i in range(1, len(word) + 1):
                self.word_prefixes.add(word[:i])
            
            # Adicionar sufixos (final da palavra)
            for i in range(1, len(word) + 1):
                self.word_suffixes.add(word[-i:])
        
        # Criar pares de palavras comuns
        for word1 in common_words[:30]:  # Usar apenas as 30 palavras mais comuns
            for word2 in common_words[:30]:
                if len(word1) > 2 and len(word2) > 2:  # Apenas palavras com mais de 2 letras
                    self.common_word_pairs.add(word1 + word2)
        
        print(f"Dicionário mínimo criado com {len(self.words)} palavras comuns.")
    
    def contains(self, word: str) -> bool:
        """
        Verifica se uma palavra está no dicionário.
        
        Args:
            word: Palavra a verificar
            
        Returns:
            True se a palavra estiver no dicionário, False caso contrário
        """
        return word.upper() in self.words
    
    def is_prefix(self, prefix: str) -> bool:
        """
        Verifica se um prefixo corresponde a alguma palavra no dicionário.
        
        Args:
            prefix: Prefixo a verificar
            
        Returns:
            True se o prefixo corresponder a alguma palavra, False caso contrário
        """
        return prefix.upper() in self.word_prefixes
    
    def is_suffix(self, suffix: str) -> bool:
        """
        Verifica se um sufixo corresponde a alguma palavra no dicionário.
        
        Args:
            suffix: Sufixo a verificar
            
        Returns:
            True se o sufixo corresponder a alguma palavra, False caso contrário
        """
        return suffix.upper() in self.word_suffixes
    
    def is_common_word_pair(self, pair: str) -> bool:
        """
        Verifica se um par de palavras é comum.
        
        Args:
            pair: Par de palavras a verificar
            
        Returns:
            True se o par for comum, False caso contrário
        """
        return pair.upper() in self.common_word_pairs
    
    def segment_text(self, text: str) -> str:
        """
        Segmenta um texto sem espaços em palavras usando o dicionário.
        
        Args:
            text: Texto sem espaços
            
        Returns:
            Texto com espaços entre palavras
        """
        # Converter para maiúsculas
        text = text.upper()
        
        # Programação dinâmica para encontrar a melhor segmentação
        n = len(text)
        
        # best_segmentation[i] = melhor segmentação até a posição i
        best_segmentation = [None] * (n + 1)
        best_segmentation[0] = []
        
        # best_score[i] = pontuação da melhor segmentação até a posição i
        best_score = [float('-inf')] * (n + 1)
        best_score[0] = 0
        
        # Lista de palavras de uma letra válidas em português
        valid_single_letters = ["A", "E", "O"]
        
        for i in range(1, n + 1):
            for j in range(max(0, i - 15), i):  # Limitar o tamanho máximo da palavra para 15
                word = text[j:i]
                
                # Calcular pontuação para esta palavra
                word_score = 0
                
                # Verificar se é uma palavra válida
                if self.contains(word):
                    word_score = len(word) ** 2  # Palavras mais longas têm pontuação maior
                
                # Verificar se é uma letra única válida (apenas A, E, O)
                elif len(word) == 1 and word in valid_single_letters:
                    word_score = 0.5  # Pontuação baixa mas positiva para letras válidas
                
                # Verificar se é um prefixo ou sufixo de palavra válida
                elif self.is_prefix(word) or self.is_suffix(word):
                    word_score = len(word) * 0.5  # Metade da pontuação para prefixos/sufixos
                
                # Verificar se é um par comum de palavras
                elif len(word) > 4 and any(self.is_common_word_pair(word[k:k+len(word)-k]) for k in range(1, len(word)-1)):
                    word_score = len(word) * 0.75  # 75% da pontuação para pares comuns
                
                # Penalizar palavras não reconhecidas
                elif len(word) == 1:
                    word_score = -1.0  # Penalizar letras isoladas que não são A, E, O
                else:
                    word_score = -len(word)  # Penalizar palavras não reconhecidas
                
                # Verificar se esta segmentação é melhor
                score = best_score[j] + word_score
                if score > best_score[i]:
                    best_score[i] = score
                    best_segmentation[i] = best_segmentation[j] + [word]
        
        # Retornar a melhor segmentação
        return ' '.join(best_segmentation[n])
    
    def count_valid_words(self, text: str) -> Tuple[int, int]:
        """
        Conta quantas palavras do texto estão no dicionário.
        
        Args:
            text: Texto para analisar
            
        Returns:
            Tupla (número de palavras válidas, número total de palavras)
        """
        # Segmentar o texto primeiro
        segmented_text = self.segment_text(text[:200])  # Limitar para os primeiros 200 caracteres
        
        # Extrair palavras
        words = segmented_text.split()
        
        if not words:
            return 0, 0
        
        # Contar palavras válidas
        valid_count = sum(1 for word in words if self.contains(word))
        
        return valid_count, len(words)
# Modelo de linguagem avançado
class LanguageModel:
    """Modelo de linguagem avançado para pontuação de textos."""
    
    def __init__(self, ngram_cache_path: str = "ngram_cache.pkl"):
        """
        Inicializa o modelo de linguagem.
        
        Args:
            ngram_cache_path: Caminho para o cache de n-gramas
        """
        self.ngram_cache_path = ngram_cache_path
        self.ngram_log_probs = {}
        self.load_or_create_ngram_model()
    
    def load_or_create_ngram_model(self):
        """Carrega ou cria o modelo de n-gramas."""
        if os.path.exists(self.ngram_cache_path):
            try:
                with open(self.ngram_cache_path, 'rb') as f:
                    self.ngram_log_probs = pickle.load(f)
                print(f"Modelo de n-gramas carregado com {len(self.ngram_log_probs)} entradas.")
            except Exception as e:
                print(f"Erro ao carregar modelo de n-gramas: {e}")
                self.create_ngram_model()
        else:
            self.create_ngram_model()
    
    def create_ngram_model(self):
        """Cria um modelo de n-gramas básico."""
        # Criar um modelo básico com bigramas e trigramas comuns
        self.ngram_log_probs = {}
        
        # Adicionar bigramas comuns
        for i, bigram in enumerate(COMMON_BIGRAMS):
            self.ngram_log_probs[bigram] = math.log10(1.0 - (i * 0.01))
        
        # Adicionar trigramas comuns
        for i, trigram in enumerate(COMMON_TRIGRAMS):
            self.ngram_log_probs[trigram] = math.log10(1.0 - (i * 0.01))
        
        # Adicionar quadrigramas comuns
        for i, quadgram in enumerate(COMMON_QUADGRAMS):
            self.ngram_log_probs[quadgram] = math.log10(1.0 - (i * 0.01))
        
        # Salvar o modelo
        try:
            with open(self.ngram_cache_path, 'wb') as f:
                pickle.dump(self.ngram_log_probs, f)
            print(f"Modelo de n-gramas básico criado com {len(self.ngram_log_probs)} entradas.")
        except Exception as e:
            print(f"Erro ao salvar modelo de n-gramas: {e}")
    
    def score_text(self, text: str) -> float:
        """
        Calcula um score para o texto baseado no modelo de linguagem.
        
        Args:
            text: Texto para avaliar
            
        Returns:
            Score do texto (maior é melhor)
        """
        # Converter para maiúsculas e remover caracteres não alfabéticos
        text = re.sub(r'[^A-Za-z]', '', text.upper())
        
        if len(text) < 2:
            return -float('inf')
        
        score = 0.0
        
        # Pontuar bigramas
        for i in range(len(text) - 1):
            bigram = text[i:i+2]
            if bigram in self.ngram_log_probs:
                score += self.ngram_log_probs[bigram]
            else:
                score -= 10.0  # Penalidade para bigramas desconhecidos
        
        # Pontuar trigramas
        if len(text) >= 3:
            for i in range(len(text) - 2):
                trigram = text[i:i+3]
                if trigram in self.ngram_log_probs:
                    score += 2 * self.ngram_log_probs[trigram]  # Peso maior para trigramas
        
        # Pontuar quadrigramas
        if len(text) >= 4:
            for i in range(len(text) - 3):
                quadgram = text[i:i+4]
                if quadgram in self.ngram_log_probs:
                    score += 4 * self.ngram_log_probs[quadgram]  # Peso ainda maior para quadrigramas
        
        # Normalizar pelo tamanho do texto
        return score / len(text)

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
    # Verificar cache
    cache_key = (size, mod, limit)
    if cache_key in INVERTIBLE_MATRICES_CACHE:
        return INVERTIBLE_MATRICES_CACHE[cache_key]
    
    if size not in [2, 3]:
        raise ValueError("Apenas matrizes 2x2 e 3x3 são suportadas para geração completa")
    
    # Para matrizes 2x2, usar abordagem otimizada
    if size == 2:
        matrices = []
        count = 0
        
        # Valores de determinante que são coprimos com 26
        valid_dets = [d for d in range(1, mod) if gcd(d, mod) == 1]
        
        # Gerar matrizes com determinantes válidos
        for a in range(mod):
            for b in range(mod):
                for c in range(mod):
                    # Calcular d para obter o determinante desejado
                    for det in valid_dets:
                        d = (det + b * c) % mod
                        if (a * d) % mod == det:
                            matrix = np.array([[a, b], [c, d]])
                            matrices.append(matrix)
                            count += 1
                            
                            if limit and count >= limit:
                                INVERTIBLE_MATRICES_CACHE[cache_key] = matrices
                                return matrices
        
        INVERTIBLE_MATRICES_CACHE[cache_key] = matrices
        return matrices
    
    # Para matrizes 3x3, usar uma abordagem mais seletiva
    matrices = []
    count = 0
    
    # Valores de determinante que são coprimos com 26
    valid_dets = [d for d in range(1, mod) if gcd(d, mod) == 1]
    
    # Gerar matrizes com estruturas específicas
    for det_val in valid_dets:
        # Gerar matrizes com estrutura triangular superior
        for a in range(1, mod):
            if gcd(a, mod) == 1:  # a deve ser coprimo com 26
                for b in range(mod):
                    for c in range(mod):
                        for d in range(1, mod):
                            if gcd(d, mod) == 1:  # d deve ser coprimo com 26
                                for e in range(mod):
                                    for f in range(mod):
                                        # Calcular g para obter o determinante desejado
                                        g = (det_val * mod_inverse((a * d), mod)) % mod
                                        if gcd(g, mod) == 1:  # g deve ser coprimo com 26
                                            matrix = np.array([
                                                [a, b, c],
                                                [0, d, e],
                                                [0, 0, g]
                                            ])
                                            matrices.append(matrix)
                                            count += 1
                                            
                                            if limit and count >= limit:
                                                INVERTIBLE_MATRICES_CACHE[cache_key] = matrices
                                                return matrices
    
    # Se não tivermos matrizes suficientes, gerar aleatoriamente
    while count < (limit or 1000):
        matrix = np.random.randint(0, mod, (size, size))
        if is_invertible_matrix(matrix, mod):
            matrices.append(matrix)
            count += 1
    
    INVERTIBLE_MATRICES_CACHE[cache_key] = matrices
    return matrices

def process_chunk(args):
    """
    Processa um chunk de matrizes para força bruta paralela.
    
    Args:
        args: Tupla (chunk, ciphertext, language_model, dictionary, known_text_path)
        
    Returns:
        Lista de resultados para este chunk
    """
    chunk, ciphertext, language_model, dictionary, known_text_path = args
    results = []
    
    # Usar um conjunto para rastrear textos decifrados já processados
    # para evitar processamento duplicado
    processed_texts = set()
    
    # Contador para liberar memória periodicamente
    counter = 0
    
    for matrix in chunk:
        try:
            # Liberar memória a cada 1000 matrizes processadas
            counter += 1
            if counter % 1000 == 0:
                # Forçar coleta de lixo para liberar memória
                import gc
                gc.collect()
            
            decrypted = decrypt_hill(ciphertext, matrix)
            
            # Verificar se já processamos este texto decifrado
            decrypted_key = decrypted[:100]  # Usar os primeiros 100 caracteres como chave
            if decrypted_key in processed_texts:
                continue
            
            processed_texts.add(decrypted_key)
            
            # Calcular score
            score = 0
            if language_model:
                score = language_model.score_text(decrypted)
            
            # Adicionar pontuação de palavras válidas se tiver dicionário
            if dictionary:
                valid_count, total_count = dictionary.count_valid_words(decrypted)
                if total_count > 0:
                    word_score = valid_count / total_count
                    score += 5 * word_score  # Peso alto para palavras válidas
            
            # Verificar similaridade com texto conhecido, se disponível
            if known_text_path and os.path.exists(known_text_path):
                similarity = verify_similarity_with_known_text(decrypted, known_text_path)
                score += similarity * 50  # Peso muito alto para similaridade
            
            results.append((matrix, decrypted, score))
        except ValueError:
            continue
        except Exception as e:
            # Capturar outras exceções para evitar falha do processo
            print(f"Erro ao processar matriz: {e}")
            continue
    
    # Limpar conjunto para liberar memória antes de retornar
    processed_texts.clear()
    
    return results

def brute_force_hill_parallel(ciphertext: str, matrix_size: int, language_model=None, dictionary=None, num_threads: int = None, known_text_path: str = None) -> List[Tuple[np.ndarray, str, float]]:
    """
    Implementa força bruta paralela para quebrar a cifra de Hill.
    
    Args:
        ciphertext: Texto cifrado
        matrix_size: Tamanho da matriz (2 ou 3)
        language_model: Modelo de linguagem para pontuação (opcional)
        dictionary: Dicionário para validação de palavras (opcional)
        num_threads: Número de threads paralelas (opcional)
        known_text_path: Caminho para o arquivo de texto conhecido (opcional)
        
    Returns:
        Lista de tuplas (matriz_chave, texto_decifrado, score) ordenadas por score
        
    Raises:
        ValueError: Se o tamanho da matriz não for suportado
    """
    if matrix_size not in [2, 3]:
        raise ValueError("Apenas matrizes 2x2 e 3x3 são suportadas para força bruta")
    
    # Determinar número de threads
    if not num_threads:
        # Limitar o número de threads para evitar uso excessivo de memória
        num_threads = min(8, mp.cpu_count())
    
    # Configurar logging para o processo
    log_file = f"hill_breaker_{matrix_size}x{matrix_size}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Iniciando quebra de cifra {matrix_size}x{matrix_size}")
    
    try:
        # Gerar matrizes inversíveis
        logging.info("Gerando matrizes inversíveis...")
        
        # Limitar o número de matrizes para 3x3 para evitar uso excessivo de memória
        limit = None
        if matrix_size == 3:
            limit = 10000  # Limitar a 10.000 matrizes para 3x3
            
        matrices = generate_invertible_matrices(matrix_size, limit=limit)
        logging.info(f"Geradas {len(matrices)} matrizes inversíveis")
        
        # Dividir matrizes em chunks menores para processamento em lotes
        # Usar chunks menores para melhor gerenciamento de memória
        chunk_size = 500 if matrix_size == 3 else 2000
        num_chunks = (len(matrices) + chunk_size - 1) // chunk_size
        chunks = [matrices[i:i+chunk_size] for i in range(0, len(matrices), chunk_size)]
        logging.info(f"Dividido em {num_chunks} chunks de tamanho {chunk_size}")
        
        # Processar em paralelo usando ThreadPoolExecutor
        results = []
        best_results = []  # Manter apenas os melhores resultados
        
        logging.info(f"Iniciando processamento paralelo com {num_threads} threads")
        
        # Processar chunks em lotes para gerenciar memória
        for batch_idx, batch_chunks in enumerate(chunks_to_batches(chunks, 5)):  # 5 chunks por lote
            logging.info(f"Processando lote {batch_idx+1}/{(num_chunks+4)//5}")
            
            batch_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Preparar argumentos para a função de processamento
                args = [(chunk, ciphertext, language_model, dictionary, known_text_path) for chunk in batch_chunks]
                
                # Usar submit e as_completed para melhor controle e feedback
                future_to_chunk = {executor.submit(process_chunk, arg): i for i, arg in enumerate(args)}
                
                # Processar resultados à medida que são concluídos
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk_index = future_to_chunk[future]
                    try:
                        chunk_result = future.result()
                        batch_results.extend(chunk_result)
                        # Feedback de progresso
                        logging.info(f"Processado chunk {chunk_index + 1}/{len(batch_chunks)} ({len(chunk_result)} resultados)")
                        print(f"Processado chunk {chunk_index + 1}/{len(batch_chunks)} ({len(chunk_result)} resultados)")
                    except Exception as e:
                        logging.error(f"Erro ao processar chunk {chunk_index}: {e}")
                        print(f"Erro ao processar chunk {chunk_index}: {e}")
            
            # Ordenar resultados do lote por score
            batch_results.sort(key=lambda x: x[2], reverse=True)
            
            # Manter apenas os 100 melhores resultados de cada lote
            best_results.extend(batch_results[:100])
            
            # Ordenar e limitar os melhores resultados globais
            best_results.sort(key=lambda x: x[2], reverse=True)
            best_results = best_results[:200]  # Manter apenas os 200 melhores resultados globais
            
            # Salvar checkpoint após cada lote
            save_checkpoint(best_results, matrix_size)
            
            # Liberar memória
            batch_results.clear()
            import gc
            gc.collect()
        
        logging.info(f"Processamento concluído. {len(best_results)} resultados encontrados.")
        return best_results
        
    except Exception as e:
        logging.error(f"Erro durante a quebra da cifra: {e}")
        # Tentar carregar o último checkpoint em caso de erro
        checkpoint_results = load_checkpoint(matrix_size)
        if checkpoint_results:
            logging.info(f"Carregado checkpoint com {len(checkpoint_results)} resultados")
            return checkpoint_results
        raise

def chunks_to_batches(chunks, batch_size):
    """
    Divide uma lista de chunks em lotes.
    
    Args:
        chunks: Lista de chunks
        batch_size: Tamanho do lote
        
    Returns:
        Lista de lotes (cada lote é uma lista de chunks)
    """
    for i in range(0, len(chunks), batch_size):
        yield chunks[i:i+batch_size]

def save_checkpoint(results, matrix_size):
    """
    Salva um checkpoint dos resultados.
    
    Args:
        results: Lista de resultados
        matrix_size: Tamanho da matriz
    """
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{matrix_size}x{matrix_size}.pkl")
    
    try:
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(results, f)
        logging.info(f"Checkpoint salvo em {checkpoint_path}")
    except Exception as e:
        logging.error(f"Erro ao salvar checkpoint: {e}")

def load_checkpoint(matrix_size):
    """
    Carrega um checkpoint dos resultados.
    
    Args:
        matrix_size: Tamanho da matriz
        
    Returns:
        Lista de resultados ou None se o checkpoint não existir
    """
    checkpoint_path = os.path.join("checkpoints", f"checkpoint_{matrix_size}x{matrix_size}.pkl")
    
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        with open(checkpoint_path, 'rb') as f:
            results = pickle.load(f)
        logging.info(f"Checkpoint carregado de {checkpoint_path}")
        return results
    except Exception as e:
        logging.error(f"Erro ao carregar checkpoint: {e}")
        return None
    
    # Determinar número de processos
    if not num_processes:
        num_processes = mp.cpu_count()
    
    # Gerar matrizes inversíveis
    matrices = generate_invertible_matrices(matrix_size)
    
    # Dividir matrizes entre processos
    chunks = np.array_split(matrices, num_processes)
    
    # Preparar argumentos para a função de processamento
    args = [(chunk, ciphertext, language_model, dictionary) for chunk in chunks]
    
    # Processar em paralelo usando ThreadPoolExecutor em vez de ProcessPoolExecutor
    # para evitar problemas de serialização
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
        chunk_results = list(executor.map(process_chunk, args))
        for chunk_result in chunk_results:
            results.extend(chunk_result)
    
    # Ordenar por score
    results.sort(key=lambda x: x[2], reverse=True)
    
    return results
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
    
    # Estratégia 1: Usar blocos de matrizes 2x2 conhecidas
    if matrix_size == 4:
        # Gerar algumas matrizes 2x2 inversíveis
        small_matrices = generate_invertible_matrices(2, mod, 20)
        
        # Criar matrizes 4x4 a partir de blocos 2x2
        for _ in range(limit // 2):
            block1 = random.choice(small_matrices)
            block2 = random.choice(small_matrices)
            block3 = random.choice(small_matrices)
            block4 = random.choice(small_matrices)
            
            matrix = np.block([[block1, block2], [block3, block4]])
            
            # Verificar se é inversível
            if is_invertible_matrix(matrix, mod):
                matrices.append(matrix)
                count += 1
    
    # Estratégia 2: Para 5x5, usar uma estrutura em blocos
    elif matrix_size == 5:
        # Gerar algumas matrizes 2x2 e 3x3 inversíveis
        matrices_2x2 = generate_invertible_matrices(2, mod, 10)
        matrices_3x3 = generate_invertible_matrices(3, mod, 10)
        
        for _ in range(limit // 2):
            # Criar matriz 5x5 com blocos
            block1 = random.choice(matrices_2x2)
            block2 = np.random.randint(0, mod, (2, 3))
            block3 = np.random.randint(0, mod, (3, 2))
            block4 = random.choice(matrices_3x3)
            
            matrix = np.block([[block1, block2], [block3, block4]])
            
            # Verificar se é inversível
            if is_invertible_matrix(matrix, mod):
                matrices.append(matrix)
                count += 1
    
    # Estratégia 3: Gerar matrizes com estrutura específica
    for _ in range(limit - count):
        # Criar matriz com diagonal principal não-nula
        matrix = np.zeros((matrix_size, matrix_size), dtype=int)
        
        # Preencher diagonal principal com valores coprimos com 26
        valid_values = [v for v in range(1, mod) if gcd(v, mod) == 1]
        for i in range(matrix_size):
            matrix[i, i] = random.choice(valid_values)
        
        # Preencher o resto da matriz com valores aleatórios
        for i in range(matrix_size):
            for j in range(matrix_size):
                if i != j:
                    matrix[i, j] = np.random.randint(0, mod)
        
        # Verificar se é inversível
        if is_invertible_matrix(matrix, mod):
            matrices.append(matrix)
            count += 1
            
            if count >= limit:
                break
    
    return matrices

# Estratégias de pontuação (Scoring) para matrizes maiores
class ScoringStrategy:
    """Classe para implementar estratégias de pontuação para candidatos."""
    
    def __init__(self, language_model=None, dictionary=None):
        """
        Inicializa a estratégia de pontuação.
        
        Args:
            language_model: Modelo de linguagem (opcional)
            dictionary: Dicionário de palavras (opcional)
        """
        self.language_model = language_model
        self.dictionary = dictionary
    
    def score_candidate(self, decrypted_text: str) -> float:
        """
        Calcula um score para o texto decifrado.
        
        Args:
            decrypted_text: Texto decifrado
            
        Returns:
            Score do texto (maior é melhor)
        """
        score = 0
        
        # 1. Pontuação baseada no modelo de linguagem
        if self.language_model:
            score += 2 * self.language_model.score_text(decrypted_text)
        
        # 2. Pontuação baseada em n-gramas comuns
        bigram_score = score_ngrams(decrypted_text, COMMON_BIGRAMS, 2)
        trigram_score = score_ngrams(decrypted_text, COMMON_TRIGRAMS, 3)
        quadgram_score = score_ngrams(decrypted_text, COMMON_QUADGRAMS, 4)
        
        # Pesos para diferentes componentes do score
        score += 5 * bigram_score + 10 * trigram_score + 20 * quadgram_score
        
        # 3. Pontuação baseada no dicionário
        if self.dictionary:
            valid_count, total_count = self.dictionary.count_valid_words(decrypted_text)
            if total_count > 0:
                word_score = valid_count / total_count
                score += 30 * word_score  # Peso alto para palavras válidas
        
        # 4. Penalizar sequências improváveis
        unlikely_patterns = ['ZZ', 'QQ', 'JJ', 'XX', 'WW', 'KK', 'YY']
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
# Implementação de técnicas avançadas para matrizes 4x4 e 5x5
class AdvancedHillBreaker:
    """Implementa técnicas avançadas para quebrar cifras Hill com matrizes grandes."""
    
    def __init__(self, matrix_size: int, language_model=None, dictionary=None, num_threads: int = None):
        """
        Inicializa o quebrador avançado.
        
        Args:
            matrix_size: Tamanho da matriz (4 ou 5)
            language_model: Modelo de linguagem (opcional)
            dictionary: Dicionário de palavras (opcional)
            num_threads: Número de threads paralelas (opcional)
        """
        self.matrix_size = matrix_size
        self.language_model = language_model
        self.dictionary = dictionary
        self.num_threads = num_threads or min(32, mp.cpu_count() * 2)  # Usar mais threads que CPUs
        self.scoring_strategy = ScoringStrategy(language_model, dictionary)
        self.shutter_heuristic = ShutterHeuristic(matrix_size)
        self.best_candidates = []
        self.iteration = 0
        self.max_iterations = 100  # 100 iterações
    
    def break_cipher(self, ciphertext: str, known_plaintext: str = None, known_text_path: str = None) -> List[Tuple[np.ndarray, str, float]]:
        """
        Tenta quebrar a cifra usando técnicas avançadas.
        
        Args:
            ciphertext: Texto cifrado
            known_plaintext: Texto claro conhecido (opcional)
            known_text_path: Caminho para o arquivo de texto conhecido original (opcional)
            
        Returns:
            Lista de tuplas (matriz_chave, texto_decifrado, score) ordenadas por score
        """
        # Se tiver texto conhecido, usar ataque com texto conhecido
        if known_plaintext:
            try:
                key_matrices = known_plaintext_attack(known_plaintext, ciphertext, self.matrix_size)
                results = []
                for key_matrix in key_matrices:
                    try:
                        decrypted = decrypt_hill(ciphertext, key_matrix)
                        score = self.scoring_strategy.score_candidate(decrypted)
                        
                        # Verificar similaridade com texto conhecido, se disponível
                        if known_text_path and os.path.exists(known_text_path):
                            similarity = verify_similarity_with_known_text(decrypted, known_text_path)
                            score += similarity * 50  # Peso muito alto para similaridade
                        
                        results.append((key_matrix, decrypted, score))
                    except ValueError:
                        continue
                
                if results:
                    return sorted(results, key=lambda x: x[2], reverse=True)
                else:
                    print("Ataque com texto conhecido falhou, tentando outras abordagens...")
            except ValueError:
                print("Ataque com texto conhecido falhou, tentando outras abordagens...")
        
        # Analisar padrões no texto cifrado
        pattern_distances = pattern_analysis(ciphertext, self.matrix_size)
        pattern_scores = analyze_pattern_distances(pattern_distances, self.matrix_size)
        
        # Inicializar com algumas matrizes promissoras
        candidates = generate_promising_matrices(self.matrix_size, limit=100)  # 100 candidatos
        
        # Processo iterativo de refinamento
        results = []
        
        for self.iteration in range(self.max_iterations):
            print(f"Iteração {self.iteration + 1}/{self.max_iterations}")
            
            # Avaliar candidatos atuais em paralelo usando ThreadPoolExecutor
            iteration_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                future_to_matrix = {
                    executor.submit(self.evaluate_candidate, matrix, ciphertext, known_text_path): matrix 
                    for matrix in candidates
                }
                for future in concurrent.futures.as_completed(future_to_matrix):
                    result = future.result()
                    if result:
                        iteration_results.append(result)
            
            # Atualizar melhores resultados
            results.extend(iteration_results)
            results.sort(key=lambda x: x[2], reverse=True)
            results = results[:100]  # Manter apenas os 100 melhores
            
            # Mostrar o melhor resultado atual
            if results:
                best_matrix, best_text, best_score = results[0]
                print(f"  Melhor score atual: {best_score:.4f}")
                print(f"  Texto: {best_text[:50]}..." if len(best_text) > 50 else best_text)
            
            # Atualizar heurística de janela
            self.shutter_heuristic.update_regions(iteration_results)
            
            # Gerar novos candidatos baseados nas regiões promissoras
            candidates = self.shutter_heuristic.generate_candidates(num_candidates=100)
            
            # Verificar critério de parada
            if self.iteration > 20 and results and results[0][2] > 0:
                # Se temos um bom candidato após 20 iterações, podemos parar
                print("Encontrado candidato com score positivo, parando busca.")
                break
        
        return results
    
    def evaluate_candidate(self, matrix: np.ndarray, ciphertext: str, known_text_path: str = None) -> Optional[Tuple[np.ndarray, str, float]]:
        """
        Avalia um candidato.
        
        Args:
            matrix: Matriz candidata
            ciphertext: Texto cifrado
            known_text_path: Caminho para o arquivo de texto conhecido (opcional)
            
        Returns:
            Tupla (matriz, texto_decifrado, score) ou None se falhar
        """
        try:
            decrypted = decrypt_hill(ciphertext, matrix)
            score = self.scoring_strategy.score_candidate(decrypted)
            
            # Verificar similaridade com texto conhecido, se disponível
            if known_text_path and os.path.exists(known_text_path):
                similarity = verify_similarity_with_known_text(decrypted, known_text_path)
                score += similarity * 50  # Peso muito alto para similaridade
            
            return (matrix, decrypted, score)
        except ValueError:
            return None
        results = []
        
        for self.iteration in range(self.max_iterations):
            print(f"Iteração {self.iteration + 1}/{self.max_iterations}")
            
            # Avaliar candidatos atuais em paralelo usando ThreadPoolExecutor
            iteration_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                future_to_matrix = {
                    executor.submit(self.evaluate_candidate, matrix, ciphertext): matrix 
                    for matrix in candidates
                }
                for future in concurrent.futures.as_completed(future_to_matrix):
                    result = future.result()
                    if result:
                        iteration_results.append(result)
            
            # Atualizar melhores resultados
            results.extend(iteration_results)
            results.sort(key=lambda x: x[2], reverse=True)
            results = results[:100]  # Manter apenas os 100 melhores
            
            # Atualizar heurística de janela
            self.shutter_heuristic.update_regions(iteration_results)
            
            # Gerar novos candidatos baseados nas regiões promissoras
            candidates = self.shutter_heuristic.generate_candidates(num_candidates=100)
            
            # Verificar critério de parada
            if self.iteration > 20 and results and results[0][2] > 0:
                # Se temos um bom candidato após 20 iterações, podemos parar
                break
        
        return results
    
    def evaluate_candidate(self, matrix: np.ndarray, ciphertext: str, known_text_path: str = None) -> Optional[Tuple[np.ndarray, str, float]]:
        """
        Avalia um candidato.
        
        Args:
            matrix: Matriz candidata
            ciphertext: Texto cifrado
            known_text_path: Caminho para o arquivo de texto conhecido (opcional)
            
        Returns:
            Tupla (matriz, texto_decifrado, score) ou None se falhar
        """
        try:
            decrypted = decrypt_hill(ciphertext, matrix)
            score = self.scoring_strategy.score_candidate(decrypted)
            return (matrix, decrypted, score)
        except ValueError:
            return None

# Pós-processamento do texto decifrado
def post_process_text(text: str, dictionary: PortugueseDictionary = None) -> str:
    """
    Realiza pós-processamento no texto decifrado para melhorar a legibilidade.
    
    Args:
        text: Texto decifrado
        dictionary: Dicionário para segmentação de palavras (opcional)
        
    Returns:
        Texto processado com espaços e pontuação
    """
    # Converter para maiúsculas e remover caracteres não alfabéticos
    text = re.sub(r'[^A-Za-z]', '', text.upper())
    
    # Se tiver dicionário, usar para segmentar o texto
    if dictionary:
        text = dictionary.segment_text(text)
    else:
        # Segmentação simples baseada em n-gramas comuns
        text = segment_text(text)
    
    # Adicionar pontuação básica (heurística simples)
    # Adicionar ponto final a cada 10-15 palavras
    words = text.split()
    sentences = []
    for i in range(0, len(words), random.randint(10, 15)):
        sentence = ' '.join(words[i:i+random.randint(10, 15)])
        if sentence:
            sentences.append(sentence.capitalize() + '.')
    
    return ' '.join(sentences)

# Classe principal para decifrar a Cifra de Hill
class HillCipherBreaker:
    """Classe principal para decifrar a Cifra de Hill."""
    
    def __init__(self, dictionary_path: str = DICT_PATH):
        """
        Inicializa o quebrador de cifra.
        
        Args:
            dictionary_path: Caminho para o arquivo de dicionário
        """
        self.dictionary = PortugueseDictionary(dictionary_path)
        self.language_model = LanguageModel()
        self.num_threads = min(32, mp.cpu_count() * 2)  # Usar mais threads que CPUs
        print(f"Usando {self.num_threads} threads para processamento paralelo.")
    
    def break_cipher(self, ciphertext: str, matrix_size: int, known_plaintext: str = None, known_text_path: str = None) -> List[Tuple[np.ndarray, str, float]]:
        """
        Tenta quebrar a cifra usando vários métodos.
        
        Args:
            ciphertext: Texto cifrado
            matrix_size: Tamanho da matriz
            known_plaintext: Texto claro conhecido (opcional)
            known_text_path: Caminho para o arquivo de texto conhecido original (opcional)
            
        Returns:
            Lista de tuplas (matriz_chave, texto_decifrado, score) ordenadas por score
        """
        results = []
        
        # 1. Se tiver texto conhecido, usar ataque com texto conhecido
        if known_plaintext:
            try:
                key_matrices = known_plaintext_attack(known_plaintext, ciphertext, matrix_size)
                for key_matrix in key_matrices:
                    try:
                        decrypted = decrypt_hill(ciphertext, key_matrix)
                        score = self.language_model.score_text(decrypted)
                        
                        # Se tiver caminho para texto conhecido, verificar similaridade
                        if known_text_path and os.path.exists(known_text_path):
                            similarity = verify_similarity_with_known_text(decrypted, known_text_path)
                            # Aumentar significativamente o score se houver alta similaridade
                            score += similarity * 50  # Peso alto para similaridade
                        
                        results.append((key_matrix, decrypted, score))
                    except ValueError:
                        continue
            except ValueError as e:
                print(f"Erro no ataque com texto conhecido: {e}")
        
        # 2. Para matrizes pequenas, usar força bruta paralela
        if matrix_size in [2, 3]:
            try:
                print(f"Iniciando força bruta paralela para matriz {matrix_size}x{matrix_size}...")
                brute_force_results = brute_force_hill_parallel(
                    ciphertext, matrix_size, self.language_model, self.dictionary, 
                    self.num_threads, known_text_path
                )
                results.extend(brute_force_results)
            except ValueError as e:
                print(f"Erro na força bruta: {e}")
        
        # 3. Para matrizes maiores, usar técnicas avançadas
        elif matrix_size in [4, 5]:
            try:
                print(f"Iniciando técnicas avançadas para matriz {matrix_size}x{matrix_size}...")
                advanced_breaker = AdvancedHillBreaker(
                    matrix_size, self.language_model, self.dictionary, self.num_threads
                )
                advanced_results = advanced_breaker.break_cipher(ciphertext, known_plaintext, known_text_path)
                results.extend(advanced_results)
            except Exception as e:
                print(f"Erro nas técnicas avançadas: {e}")
        
        # Ordenar resultados por score
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results
    
    def generate_report(self, results: List[Tuple[np.ndarray, str, float]], ciphertext: str, matrix_size: int, known_text_path: str = None) -> str:
        """
        Gera um relatório dos resultados.
        
        Args:
            results: Lista de resultados (matriz_chave, texto_decifrado, score)
            ciphertext: Texto cifrado original
            matrix_size: Tamanho da matriz
            known_text_path: Caminho para o arquivo de texto conhecido original (opcional)
            
        Returns:
            Relatório formatado
        """
        report = []
        report.append("=== RELATÓRIO DE DECIFRAGEM DA CIFRA DE HILL (OTIMIZADO) ===")
        report.append(f"Tamanho da matriz: {matrix_size}x{matrix_size}")
        report.append(f"Texto cifrado: {ciphertext[:50]}..." if len(ciphertext) > 50 else ciphertext)
        report.append(f"Número de resultados: {len(results)}")
        
        if results:
            report.append("\nMelhores resultados:")
            for i, (matrix, decrypted, score) in enumerate(results[:5], 1):
                report.append(f"\n--- Resultado #{i} (Score: {score:.4f}) ---")
                report.append(f"Matriz chave:\n{matrix}")
                
                # Texto decifrado original
                report.append(f"Texto decifrado (bruto): {decrypted[:100]}..." if len(decrypted) > 100 else decrypted)
                
                # Texto decifrado pós-processado
                processed_text = post_process_text(decrypted, self.dictionary)
                report.append(f"Texto decifrado (processado): {processed_text[:100]}..." if len(processed_text) > 100 else processed_text)
                
                # Análise de palavras válidas
                valid_count, total_count = self.dictionary.count_valid_words(decrypted[:200])
                if total_count > 0:
                    report.append(f"Palavras válidas: {valid_count}/{total_count} ({valid_count/total_count:.2%})")
                
                # Verificar similaridade com texto conhecido, se disponível
                if known_text_path and os.path.exists(known_text_path):
                    similarity = verify_similarity_with_known_text(decrypted, known_text_path)
                    report.append(f"Similaridade com texto conhecido: {similarity:.2%}")
        else:
            report.append("\nNenhum resultado encontrado.")
        
        return "\n".join(report)
# Função principal
def main():
    """Função principal do programa."""
    print("=== Hill Cipher Breaker (Optimized) ===")
    
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
            
            # Encontrar o texto original correspondente
            original_text_path = None
            for text_file in os.listdir(os.path.join(known_dir, "textos")):
                if text_file.endswith(".txt"):
                    original_text_path = os.path.join(known_dir, "textos", text_file)
                    break
            
            print(f"\nQuebrando cifra {size}x{size}...")
            start_time = time.time()
            results = breaker.break_cipher(ciphertext, size, known_text_path=original_text_path)
            elapsed_time = time.time() - start_time
            
            if results:
                report = breaker.generate_report(results, ciphertext, size, known_text_path=original_text_path)
                print(report)
                print(f"Tempo de execução: {elapsed_time:.2f} segundos")
                
                # Salvar relatório
                report_path = f"relatorios/otimizado/conhecidos/hill_{size}x{size}/relatorio.txt"
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
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
            results = breaker.break_cipher(ciphertext, size)  # Sem texto conhecido para comparação
            elapsed_time = time.time() - start_time
            
            if results:
                report = breaker.generate_report(results, ciphertext, size)  # Sem texto conhecido para comparação
                print(report)
                print(f"Tempo de execução: {elapsed_time:.2f} segundos")
                
                # Salvar relatório
                report_path = f"relatorios/otimizado/desconhecidos/hill_{size}x{size}/relatorio.txt"
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                with open(report_path, 'w') as f:
                    f.write(report)
                    f.write(f"\n\nTempo de execução: {elapsed_time:.2f} segundos")
                print(f"Relatório salvo em {report_path}")

if __name__ == "__main__":
    main()
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
def verify_similarity_with_known_text(decrypted_text: str, known_text_path: str) -> float:
    """
    Verifica a similaridade entre o texto decifrado e o texto conhecido original.
    
    Args:
        decrypted_text: Texto decifrado a ser verificado
        known_text_path: Caminho para o arquivo de texto conhecido
        
    Returns:
        Pontuação de similaridade (0 a 1, onde 1 é perfeita correspondência)
    """
    try:
        # Carregar o texto conhecido
        with open(known_text_path, 'r', encoding='latin-1') as f:  # Usar latin-1 em vez de utf-8
            known_text = f.read()
        
        # Normalizar os textos (remover espaços, pontuação, converter para maiúsculas)
        known_text = re.sub(r'[^A-Za-z]', '', known_text.upper())
        decrypted_text = re.sub(r'[^A-Za-z]', '', decrypted_text.upper())
        
        # Verificar se o texto decifrado está contido no texto conhecido
        if decrypted_text in known_text:
            return 1.0  # Correspondência perfeita
        
        # Verificar correspondência parcial usando múltiplos tamanhos de n-gramas
        # para uma análise mais robusta
        similarity_scores = []
        
        # Usar diferentes tamanhos de n-gramas para uma análise mais completa
        for ngram_size in [3, 4, 5, 6]:
            # Extrair n-gramas do texto decifrado
            decrypted_ngrams = [decrypted_text[i:i+ngram_size] 
                               for i in range(len(decrypted_text) - ngram_size + 1)]
            
            if not decrypted_ngrams:
                continue
            
            # Contar quantos n-gramas estão presentes no texto conhecido
            matches = sum(1 for ngram in decrypted_ngrams if ngram in known_text)
            
            # Calcular pontuação de similaridade para este tamanho de n-grama
            if matches > 0:
                # Dar mais peso para n-gramas maiores
                weight = ngram_size / 4.0  # Normalizado para que o peso de n-grama 4 seja 1.0
                similarity_scores.append((matches / len(decrypted_ngrams)) * weight)
        
        # Se não encontrou nenhuma correspondência com nenhum tamanho de n-grama
        if not similarity_scores:
            return 0.0
        
        # Retornar a média das pontuações de similaridade
        return sum(similarity_scores) / len(similarity_scores)
    
    except Exception as e:
        print(f"Erro ao verificar similaridade: {e}")
        return 0.0

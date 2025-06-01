#!/usr/bin/env python3
"""
Hill Cipher Breaker (Advanced) - Programa para decifrar a Cifra de Hill

Este programa implementa métodos avançados para quebrar a cifra de Hill,
incluindo processamento paralelo, dicionário de palavras em português,
uso de texto conhecido e modelos de linguagem sofisticados.

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
import urllib.request
import pickle
import random

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

# URL para baixar dicionário de português
DICT_URL = "https://www.ime.usp.br/~pf/dicios/br-sem-acentos.txt"
DICT_PATH = "portuguese_dict.txt"

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
    
    if not ngrams:
        return 0
    
    # Contar ocorrências de n-gramas comuns
    score = sum(1 for ngram in ngrams if ngram in common_ngrams)
    
    # Normalizar pelo tamanho do texto
    return score / len(ngrams)

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
        self.load_dictionary()
    
    def load_dictionary(self):
        """Carrega o dicionário de palavras."""
        # Verificar se o arquivo existe
        if not os.path.exists(self.dict_path):
            try:
                print(f"Baixando dicionário de português de {DICT_URL}...")
                # Ignorar verificação de certificado SSL para contornar o erro
                import ssl
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
            with open(self.dict_path, 'r', encoding='utf-8') as f:
                self.words = set(word.strip().upper() for word in f if word.strip())
            print(f"Dicionário carregado com {len(self.words)} palavras.")
        except Exception as e:
            print(f"Erro ao carregar dicionário: {e}")
            self.create_minimal_dictionary()
    
    def create_minimal_dictionary(self):
        """Cria um dicionário mínimo com palavras comuns em português."""
        common_words = [
            "A", "DE", "QUE", "O", "E", "DO", "DA", "EM", "UM", "PARA",
            "COM", "NÃO", "UMA", "OS", "NO", "SE", "NA", "POR", "MAIS", "AS",
            "DOS", "COMO", "MAS", "AO", "ELE", "DAS", "À", "SEU", "SUA", "OU",
            "QUANDO", "MUITO", "NOS", "JÁ", "EU", "TAMBÉM", "SÓ", "PELO", "PELA",
            "ATÉ", "ISSO", "ELA", "ENTRE", "DEPOIS", "SEM", "MESMO", "AOS", "SEUS",
            "QUEM", "NAS", "ME", "ESSE", "ELES", "VOCÊ", "ESSA", "NUM", "NEM", "SUAS",
            "MEU", "ÀS", "MINHA", "NUMA", "PELOS", "ELAS", "QUAL", "NESTE", "PELAS",
            "ESTE", "FOSSE", "DELE", "TU", "TE", "VOCÊS", "VOS", "LHE", "LHES", "MEUS",
            "MINHAS", "TEU", "TUA", "TEUS", "TUAS", "NOSSO", "NOSSA", "NOSSOS", "NOSSAS",
            "DELA", "DELES", "DELAS", "ESTA", "ESTES", "ESTAS", "AQUELE", "AQUELA",
            "AQUELES", "AQUELAS", "ISTO", "AQUILO", "ESTOU", "ESTÁ", "ESTAMOS", "ESTÃO",
            "ESTIVE", "ESTEVE", "ESTIVEMOS", "ESTIVERAM", "ESTAVA", "ESTÁVAMOS", "ESTAVAM",
            "ESTIVERA", "ESTIVÉRAMOS", "ESTEJA", "ESTEJAMOS", "ESTEJAM", "ESTIVESSE",
            "ESTIVÉSSEMOS", "ESTIVESSEM", "ESTIVER", "ESTIVERMOS", "ESTIVEREM"
        ]
        self.words = set(common_words)
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
    
    def count_valid_words(self, text: str) -> Tuple[int, int]:
        """
        Conta quantas palavras do texto estão no dicionário.
        
        Args:
            text: Texto para analisar
            
        Returns:
            Tupla (número de palavras válidas, número total de palavras)
        """
        # Extrair palavras do texto
        words = re.findall(r'\b[A-Za-z]{2,}\b', text.upper())
        
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
    if size not in [2, 3]:
        raise ValueError("Apenas matrizes 2x2 e 3x3 são suportadas para geração completa")
    
    # Para matrizes 2x2, podemos gerar todas as combinações
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
                                return matrices
        
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
                                                return matrices
    
    # Se não tivermos matrizes suficientes, gerar aleatoriamente
    while count < (limit or 1000):
        matrix = np.random.randint(0, mod, (size, size))
        if is_invertible_matrix(matrix, mod):
            matrices.append(matrix)
            count += 1
    
    return matrices

def process_chunk(args):
    """
    Processa um chunk de matrizes para força bruta paralela.
    
    Args:
        args: Tupla (chunk, ciphertext, language_model)
        
    Returns:
        Lista de resultados para este chunk
    """
    chunk, ciphertext, language_model = args
    results = []
    for matrix in chunk:
        try:
            decrypted = decrypt_hill(ciphertext, matrix)
            
            # Calcular score
            score = 0
            if language_model:
                score = language_model.score_text(decrypted)
            
            results.append((matrix, decrypted, score))
        except ValueError:
            continue
    return results

def brute_force_hill_parallel(ciphertext: str, matrix_size: int, language_model=None, num_processes: int = None) -> List[Tuple[np.ndarray, str, float]]:
    """
    Implementa força bruta paralela para quebrar a cifra de Hill.
    
    Args:
        ciphertext: Texto cifrado
        matrix_size: Tamanho da matriz (2 ou 3)
        language_model: Modelo de linguagem para pontuação (opcional)
        num_processes: Número de processos paralelos (opcional)
        
    Returns:
        Lista de tuplas (matriz_chave, texto_decifrado, score) ordenadas por score
        
    Raises:
        ValueError: Se o tamanho da matriz não for suportado
    """
    if matrix_size not in [2, 3]:
        raise ValueError("Apenas matrizes 2x2 e 3x3 são suportadas para força bruta")
    
    # Determinar número de processos
    if not num_processes:
        num_processes = mp.cpu_count()
    
    # Gerar matrizes inversíveis
    matrices = generate_invertible_matrices(matrix_size)
    
    # Dividir matrizes entre processos
    chunks = np.array_split(matrices, num_processes)
    
    # Preparar argumentos para a função de processamento
    args = [(chunk, ciphertext, language_model) for chunk in chunks]
    
    # Processar em paralelo
    with mp.Pool(processes=num_processes) as pool:
        all_results = pool.map(process_chunk, args)
    
    # Combinar resultados
    results = []
    for chunk_results in all_results:
        results.extend(chunk_results)
    
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
    
    def __init__(self, matrix_size: int, language_model=None, dictionary=None, num_processes: int = None):
        """
        Inicializa o quebrador avançado.
        
        Args:
            matrix_size: Tamanho da matriz (4 ou 5)
            language_model: Modelo de linguagem (opcional)
            dictionary: Dicionário de palavras (opcional)
            num_processes: Número de processos paralelos (opcional)
        """
        self.matrix_size = matrix_size
        self.language_model = language_model
        self.dictionary = dictionary
        self.num_processes = num_processes or mp.cpu_count()
        self.scoring_strategy = ScoringStrategy(language_model, dictionary)
        self.shutter_heuristic = ShutterHeuristic(matrix_size)
        self.best_candidates = []
        self.iteration = 0
        self.max_iterations = 100  # Aumentado para 100 iterações
    
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
                score = self.scoring_strategy.score_candidate(decrypted)
                return [(key_matrix, decrypted, score)]
            except ValueError:
                print("Ataque com texto conhecido falhou, tentando outras abordagens...")
        
        # Analisar padrões no texto cifrado
        pattern_distances = pattern_analysis(ciphertext, self.matrix_size)
        pattern_scores = analyze_pattern_distances(pattern_distances, self.matrix_size)
        
        # Inicializar com algumas matrizes promissoras
        candidates = generate_promising_matrices(self.matrix_size, limit=100)  # Aumentado para 100 candidatos
        
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
            candidates = self.shutter_heuristic.generate_candidates(num_candidates=100)  # Aumentado para 100 candidatos
            
            # Verificar critério de parada
            if self.iteration > 20 and results and results[0][2] > 0:
                # Se temos um bom candidato após 20 iterações, podemos parar
                break
        
        return results

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
        self.num_processes = mp.cpu_count()
        print(f"Usando {self.num_processes} processos para processamento paralelo.")
    
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
                score = self.language_model.score_text(decrypted)
                results.append((key_matrix, decrypted, score))
            except ValueError as e:
                print(f"Erro no ataque com texto conhecido: {e}")
        
        # 2. Para matrizes pequenas, usar força bruta paralela
        if matrix_size in [2, 3]:
            try:
                print(f"Iniciando força bruta paralela para matriz {matrix_size}x{matrix_size}...")
                brute_force_results = brute_force_hill_parallel(ciphertext, matrix_size, self.language_model, self.num_processes)
                results.extend(brute_force_results)
            except ValueError as e:
                print(f"Erro na força bruta: {e}")
        
        # 3. Para matrizes maiores, usar técnicas avançadas
        elif matrix_size in [4, 5]:
            try:
                print(f"Iniciando técnicas avançadas para matriz {matrix_size}x{matrix_size}...")
                advanced_breaker = AdvancedHillBreaker(matrix_size, self.language_model, self.dictionary, self.num_processes)
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
        report.append("=== RELATÓRIO DE DECIFRAGEM DA CIFRA DE HILL (AVANÇADO) ===")
        report.append(f"Tamanho da matriz: {matrix_size}x{matrix_size}")
        report.append(f"Texto cifrado: {ciphertext[:50]}..." if len(ciphertext) > 50 else ciphertext)
        report.append(f"Número de resultados: {len(results)}")
        
        if results:
            report.append("\nMelhores resultados:")
            for i, (matrix, decrypted, score) in enumerate(results[:5], 1):
                report.append(f"\n--- Resultado #{i} (Score: {score:.4f}) ---")
                report.append(f"Matriz chave:\n{matrix}")
                report.append(f"Texto decifrado: {decrypted[:100]}..." if len(decrypted) > 100 else decrypted)
                
                # Análise de palavras válidas
                if self.dictionary:
                    valid_count, total_count = self.dictionary.count_valid_words(decrypted[:200])
                    if total_count > 0:
                        report.append(f"Palavras válidas: {valid_count}/{total_count} ({valid_count/total_count:.2%})")
        else:
            report.append("\nNenhum resultado encontrado.")
        
        return "\n".join(report)
# Função principal
def main():
    """Função principal do programa."""
    print("=== Hill Cipher Breaker (Advanced) ===")
    
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
                report_path = f"relatorios/avancado/conhecidos/hill_{size}x{size}/relatorio.txt"
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
            results = breaker.break_cipher(ciphertext, size)
            elapsed_time = time.time() - start_time
            
            if results:
                report = breaker.generate_report(results, ciphertext, size)
                print(report)
                print(f"Tempo de execução: {elapsed_time:.2f} segundos")
                
                # Salvar relatório
                report_path = f"relatorios/avancado/desconhecidos/hill_{size}x{size}/relatorio.txt"
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                with open(report_path, 'w') as f:
                    f.write(report)
                    f.write(f"\n\nTempo de execução: {elapsed_time:.2f} segundos")
                print(f"Relatório salvo em {report_path}")

if __name__ == "__main__":
    main()
def evaluate_candidate(args):
    """
    Avalia um candidato para quebra de cifra.
    
    Args:
        args: Tupla (matrix, ciphertext, scoring_strategy)
        
    Returns:
        Tupla (matriz, texto_decifrado, score) ou None se falhar
    """
    matrix, ciphertext, scoring_strategy = args
    try:
        decrypted = decrypt_hill(ciphertext, matrix)
        score = scoring_strategy.score_candidate(decrypted)
        return (matrix, decrypted, score)
    except ValueError:
        return None

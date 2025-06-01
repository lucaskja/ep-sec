#!/usr/bin/env python3
"""
Enhanced Portuguese Language Model for Hill Cipher Breaker.

This module provides a more sophisticated Portuguese language model
to improve the quality of decryption results.
"""

import re
import pickle
import os
import sys
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Set

# Add the parent directory to the path so Python can find the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PortugueseLanguageModel:
    """Portuguese language model with advanced features for text analysis."""
    
    def __init__(self, dict_path: str = None):
        """
        Initialize the Portuguese language model.
        
        Args:
            dict_path: Path to Portuguese dictionary file (optional)
        """
        # Portuguese letter frequencies (more accurate)
        self.letter_freq = {
            'A': 14.63, 'E': 12.57, 'O': 10.73, 'S': 7.81, 'R': 6.53, 'I': 6.18, 
            'N': 5.05, 'D': 4.99, 'M': 4.74, 'U': 4.63, 'T': 4.34, 'C': 3.88,
            'L': 2.78, 'P': 2.52, 'V': 1.67, 'G': 1.30, 'H': 1.28, 'Q': 1.20,
            'B': 1.04, 'F': 1.02, 'Z': 0.47, 'J': 0.40, 'X': 0.21, 'K': 0.02,
            'W': 0.01, 'Y': 0.01
        }
        
        # Common Portuguese word endings
        self.common_endings = [
            'AR', 'ER', 'IR', 'OU', 'AM', 'EM', 'AO', 'OS', 'AS', 'ES', 'IS',
            'ADA', 'ADO', 'ANDO', 'ENDO', 'INDO', 'MENTE', 'IDADE', 'ACAO'
        ]
        
        # Common Portuguese word beginnings
        self.common_beginnings = [
            'DE', 'CO', 'PR', 'RE', 'IN', 'ES', 'TR', 'PE', 'DES', 'CON',
            'PRE', 'COM', 'EX', 'SUB', 'INTER', 'SUPER'
        ]
        
        # Common Portuguese words (most frequent)
        self.common_words = [
            'DE', 'A', 'O', 'QUE', 'E', 'DO', 'DA', 'EM', 'UM', 'PARA', 'COM',
            'NAO', 'UMA', 'OS', 'NO', 'SE', 'NA', 'POR', 'MAIS', 'AS', 'DOS',
            'COMO', 'MAS', 'AO', 'ELE', 'DAS', 'SEU', 'SUA', 'OU', 'QUANDO',
            'MUITO', 'NOS', 'JA', 'EU', 'TAMBEM', 'SO', 'PELO', 'PELA', 'ATE',
            'ISSO', 'ELA', 'ENTRE', 'DEPOIS', 'SEM', 'MESMO', 'AOS', 'SEUS',
            'QUEM', 'NAS', 'ME', 'ESSE', 'ELES', 'VOCE', 'ESSA'
        ]
        
        # Valid single-letter words in Portuguese
        self.valid_single_letters = ['A', 'E', 'O']
        
        # Common Portuguese bigrams
        self.common_bigrams = [
            'DE', 'RA', 'ES', 'OS', 'AR', 'QU', 'NT', 'EN', 'ER', 'RE',
            'TE', 'CO', 'OR', 'AS', 'DO', 'AD', 'TA', 'SE', 'ME', 'AN',
            'ND', 'EM', 'ED', 'PA', 'MA', 'EL', 'AM', 'AL', 'PE', 'RI'
        ]
        
        # Common Portuguese trigrams
        self.common_trigrams = [
            'QUE', 'EST', 'COM', 'NTE', 'TEM', 'ARA', 'POR', 'ENT', 'TER', 'CON',
            'RES', 'ADE', 'ERA', 'ADO', 'STA', 'PAR', 'NTO', 'AND', 'DES', 'ESS',
            'MEN', 'NDA', 'NHA', 'UMA', 'NOS', 'DOS', 'SER', 'AIS', 'ARA', 'VER'
        ]
        
        # Common Portuguese quadgrams
        self.common_quadgrams = [
            'MENT', 'ENTE', 'PARA', 'ANDO', 'AQUE', 'ESTA', 'OQUE', 'COMO', 'ADOS', 'NTES',
            'ISTA', 'IDAD', 'DESS', 'ANTE', 'ANDO', 'CONT', 'ESSE', 'NTOS', 'PRES', 'NCIA',
            'ANDO', 'INDO', 'ENDO', 'ARAM', 'ERAM', 'INHA', 'INHA', 'MENT', 'PARA', 'ENTE'
        ]
        
        # Load dictionary
        self.dictionary = set()
        self.load_dictionary_from_multiple_locations(dict_path)
        
        # Build n-gram frequency models
        self.ngram_models = {}
        self.build_ngram_models()
    
    def load_dictionary_from_multiple_locations(self, dict_path: str = None):
        """
        Try to load dictionary from multiple possible locations.
        
        Args:
            dict_path: Path to dictionary file (optional)
        """
        # List of possible dictionary locations to try
        possible_paths = []
        
        # Add the provided path if given
        if dict_path:
            possible_paths.append(dict_path)
        
        # Add common locations
        possible_paths.extend([
            "portuguese_dict.txt",                    # Project root
            "data/portuguese_dict.txt",               # Data directory
            "../portuguese_dict.txt",                 # Parent directory
            os.path.join(os.getcwd(), "portuguese_dict.txt"),  # Current working directory
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "portuguese_dict.txt")  # Project root from module
        ])
        
        # Try each path
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found dictionary at {path}")
                if self.load_dictionary_file(path):
                    return
        
        # If no dictionary was loaded, try to download it
        if not self.dictionary:
            print("Dictionary not found in any of the expected locations. Attempting to download...")
            self.download_dictionary("data/portuguese_dict.txt")
        
        # If still no dictionary, create a minimal one
        if not self.dictionary:
            print("Creating minimal dictionary with common Portuguese words.")
            self.create_minimal_dictionary()
    
    def load_dictionary_file(self, dict_path: str) -> bool:
        """
        Load dictionary from file.
        
        Args:
            dict_path: Path to dictionary file
            
        Returns:
            True if dictionary was loaded successfully, False otherwise
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(dict_path, 'r', encoding=encoding) as f:
                        for line in f:
                            word = line.strip().upper()
                            if word:
                                self.dictionary.add(word)
                    print(f"Dictionary loaded with {len(self.dictionary)} words using {encoding} encoding.")
                    return True
                except UnicodeDecodeError:
                    continue
            
            # If we get here, none of the encodings worked
            print(f"Could not decode dictionary file with any of the attempted encodings.")
            return False
            
        except Exception as e:
            print(f"Error loading dictionary from {dict_path}: {e}")
            return False
    
    def load_dictionary(self, dict_path: str):
        """
        Load Portuguese dictionary from file.
        
        Args:
            dict_path: Path to dictionary file
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(dict_path, 'r', encoding=encoding) as f:
                        for line in f:
                            word = line.strip().upper()
                            if word:
                                self.dictionary.add(word)
                    print(f"Dictionary loaded with {len(self.dictionary)} words using {encoding} encoding.")
                    break
                except UnicodeDecodeError:
                    continue
            
            # If dictionary is empty or not loaded, try to download it
            if not self.dictionary:
                print("Dictionary not loaded. Attempting to download...")
                self.download_dictionary(dict_path)
        except Exception as e:
            print(f"Error loading dictionary: {e}")
            # Create a minimal dictionary with common Portuguese words
            self.create_minimal_dictionary()
    
    def download_dictionary(self, dict_path: str):
        """
        Download Portuguese dictionary from the internet.
        
        Args:
            dict_path: Path to save the dictionary
        """
        try:
            # URLs for Portuguese dictionaries
            urls = [
                "https://www.ime.usp.br/~pf/dicios/br-utf8.txt",
                "https://raw.githubusercontent.com/pythonprobr/palavras/master/palavras.txt"
            ]
            
            for url in urls:
                try:
                    # Create SSL context that doesn't verify certificates
                    context = ssl._create_unverified_context()
                    
                    # Download dictionary
                    with urllib.request.urlopen(url, context=context) as response:
                        content = response.read().decode('utf-8')
                        
                        # Save dictionary
                        os.makedirs(os.path.dirname(dict_path), exist_ok=True)
                        with open(dict_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        # Load dictionary
                        for line in content.splitlines():
                            word = line.strip().upper()
                            if word:
                                self.dictionary.add(word)
                        
                        print(f"Dictionary downloaded and loaded with {len(self.dictionary)} words.")
                        return
                except Exception as e:
                    print(f"Error downloading dictionary from {url}: {e}")
                    continue
            
            # If all downloads fail, create a minimal dictionary
            self.create_minimal_dictionary()
        except Exception as e:
            print(f"Error downloading dictionary: {e}")
            self.create_minimal_dictionary()
    
    def create_minimal_dictionary(self):
        """Create a minimal dictionary with common Portuguese words."""
        # Add common Portuguese words
        common_words = [
            'DE', 'A', 'O', 'QUE', 'E', 'DO', 'DA', 'EM', 'UM', 'PARA', 'COM',
            'NAO', 'UMA', 'OS', 'NO', 'SE', 'NA', 'POR', 'MAIS', 'AS', 'DOS',
            'COMO', 'MAS', 'AO', 'ELE', 'DAS', 'SEU', 'SUA', 'OU', 'QUANDO',
            'MUITO', 'NOS', 'JA', 'EU', 'TAMBEM', 'SO', 'PELO', 'PELA', 'ATE',
            'ISSO', 'ELA', 'ENTRE', 'DEPOIS', 'SEM', 'MESMO', 'AOS', 'SEUS',
            'QUEM', 'NAS', 'ME', 'ESSE', 'ELES', 'VOCE', 'ESSA', 'NUM', 'NEM',
            'SUAS', 'MEU', 'MINHA', 'TEM', 'TINHA', 'FORAM', 'SAO', 'ESTAO',
            'ESTOU', 'ESTA', 'ESTAMOS', 'ESTIVE', 'ESTAVA', 'ESTAVAM', 'TEMOS',
            'TENHO', 'TINHA', 'TINHAM', 'TIVE', 'TEVE', 'TIVEMOS', 'TIVERAM',
            'CASA', 'TEMPO', 'VERDADE', 'TRABALHO', 'PARTE', 'PESSOA', 'PESSOAS',
            'HOMEM', 'MULHER', 'CRIANCA', 'VIDA', 'DIA', 'NOITE', 'AMOR', 'AGUA',
            'TERRA', 'MAR', 'PAIS', 'CIDADE', 'RUA', 'LUGAR', 'COISA', 'COISAS',
            'FORMA', 'CASO', 'PONTO', 'GRUPO', 'PROBLEMA', 'FATO', 'JEITO',
            'LADO', 'MOMENTO', 'HORA', 'SEMANA', 'MES', 'ANO', 'HOJE', 'AMANHA',
            'ONTEM', 'AGORA', 'DEPOIS', 'ANTES', 'SEMPRE', 'NUNCA', 'AQUI', 'ALI',
            'DENTRO', 'FORA', 'NOVO', 'VELHO', 'GRANDE', 'PEQUENO', 'ALTO', 'BAIXO',
            'BOM', 'MAU', 'CERTO', 'ERRADO', 'MELHOR', 'PIOR', 'PRIMEIRO', 'ULTIMO',
            'MUITOS', 'POUCOS', 'ALGUNS', 'TODAS', 'CADA', 'QUALQUER', 'NADA',
            'TUDO', 'ALGO', 'ALGUEM', 'NINGUEM', 'OUTRO', 'OUTRA', 'OUTROS', 'OUTRAS'
        ]
        
        for word in common_words:
            self.dictionary.add(word)
        
        print(f"Created minimal dictionary with {len(self.dictionary)} common Portuguese words.")
    
    def build_ngram_models(self):
        """Build n-gram frequency models from common words."""
        # Use common words to build n-gram models
        text = ' '.join(self.common_words)
        
        # Build models for n=1,2,3,4
        for n in range(1, 5):
            self.ngram_models[n] = self.extract_ngrams(text, n)
    
    def extract_ngrams(self, text: str, n: int) -> Dict[str, float]:
        """
        Extract n-grams and their frequencies from text.
        
        Args:
            text: Text to analyze
            n: Size of n-grams
            
        Returns:
            Dictionary of n-grams and their frequencies
        """
        # Clean text
        text = re.sub(r'[^A-Z]', '', text.upper())
        
        # Extract n-grams
        ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
        
        # Count occurrences
        counter = Counter(ngrams)
        
        # Calculate frequencies
        total = sum(counter.values())
        frequencies = {ngram: count / total for ngram, count in counter.items()}
        
        return frequencies
    
    def score_text(self, text: str) -> float:
        """
        Score text based on Portuguese language features.
        
        Args:
            text: Text to score
            
        Returns:
            Score indicating how likely the text is Portuguese
        """
        # Clean text
        text = re.sub(r'[^A-Z]', '', text.upper())
        
        if not text:
            return -100.0
        
        # Initialize score
        score = 0.0
        
        # Score letter frequencies
        letter_score = self.score_letter_frequencies(text)
        
        # Score n-grams
        ngram_scores = []
        for n in range(1, 5):
            if len(text) >= n:
                ngram_scores.append(self.score_ngrams(text, n))
        
        # Score word patterns
        word_score = self.score_word_patterns(text)
        
        # Combine scores with weights
        score = (
            0.2 * letter_score +
            0.5 * sum(ngram_scores) / len(ngram_scores) +
            0.3 * word_score
        )
        
        return score
    
    def score_letter_frequencies(self, text: str) -> float:
        """
        Score text based on letter frequencies.
        
        Args:
            text: Text to score
            
        Returns:
            Score based on letter frequencies
        """
        # Count letters
        counter = Counter(text)
        
        # Calculate frequencies
        total = len(text)
        observed_freq = {letter: count / total * 100 for letter, count in counter.items()}
        
        # Calculate difference from expected frequencies
        diff_sum = 0
        for letter, expected in self.letter_freq.items():
            observed = observed_freq.get(letter, 0)
            diff_sum += abs(observed - expected)
        
        # Normalize score (lower difference is better)
        max_diff = sum(self.letter_freq.values())  # Worst case
        score = 1.0 - (diff_sum / max_diff)
        
        return score
    
    def score_ngrams(self, text: str, n: int) -> float:
        """
        Score text based on n-gram frequencies.
        
        Args:
            text: Text to score
            n: Size of n-grams
            
        Returns:
            Score based on n-gram frequencies
        """
        # Extract n-grams from text
        ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
        
        if not ngrams:
            return 0.0
        
        # Get expected n-gram frequencies
        expected_freq = self.ngram_models.get(n, {})
        
        if not expected_freq:
            # Use common n-grams if model not available
            if n == 2:
                common_ngrams = self.common_bigrams
            elif n == 3:
                common_ngrams = self.common_trigrams
            elif n == 4:
                common_ngrams = self.common_quadgrams
            else:
                return 0.0
            
            # Count matches
            matches = sum(1 for ngram in ngrams if ngram in common_ngrams)
            return matches / len(ngrams)
        
        # Count matches with expected frequencies
        score = 0
        for ngram in ngrams:
            score += expected_freq.get(ngram, 0)
        
        return score / len(ngrams)
    
    def score_word_patterns(self, text: str) -> float:
        """
        Score text based on Portuguese word patterns.
        
        Args:
            text: Text to score
            
        Returns:
            Score based on word patterns
        """
        # Split text into potential words (3-15 characters)
        words = []
        for length in range(3, 16):
            for i in range(len(text) - length + 1):
                words.append(text[i:i+length])
        
        if not words:
            return 0.0
        
        # Score words
        total_score = 0
        
        for word in words:
            word_score = 0
            
            # Check if word is in dictionary
            if word in self.dictionary:
                word_score += 1.0
            
            # Check for common beginnings
            for beginning in self.common_beginnings:
                if word.startswith(beginning):
                    word_score += 0.5
                    break
            
            # Check for common endings
            for ending in self.common_endings:
                if word.endswith(ending):
                    word_score += 0.5
                    break
            
            # Check vowel-consonant pattern
            vowels = sum(1 for c in word if c in 'AEIOU')
            consonants = len(word) - vowels
            
            # Portuguese words typically have a good vowel-consonant balance
            if 0.3 <= vowels / len(word) <= 0.7:
                word_score += 0.3
            
            total_score += word_score
        
        return total_score / len(words)
    
    def contains(self, word: str) -> bool:
        """
        Check if word is in dictionary.
        
        Args:
            word: Word to check
            
        Returns:
            True if word is in dictionary
        """
        return word in self.dictionary or (len(word) == 1 and word in self.valid_single_letters)
    
    def is_prefix(self, word: str) -> bool:
        """
        Check if word is a prefix of a dictionary word.
        
        Args:
            word: Word to check
            
        Returns:
            True if word is a prefix
        """
        return any(beginning.startswith(word) for beginning in self.common_beginnings)
    
    def is_suffix(self, word: str) -> bool:
        """
        Check if word is a suffix of a dictionary word.
        
        Args:
            word: Word to check
            
        Returns:
            True if word is a suffix
        """
        return any(ending.endswith(word) for ending in self.common_endings)
    
    def count_valid_words(self, text: str) -> Tuple[int, int]:
        """
        Count valid words in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (valid word count, total word count)
        """
        # Clean text and insert spaces for better word detection
        processed_text = self.insert_spaces(text)
        
        # Split into words
        words = [word for word in processed_text.split() if word]
        
        if not words:
            return 0, 0
        
        # Count valid words
        valid_count = 0
        for word in words:
            if word in self.dictionary or (len(word) == 1 and word in self.valid_single_letters):
                valid_count += 1
                
        return valid_count, len(words)
    
    def insert_spaces(self, text: str) -> str:
        """
        Insert spaces into text to separate words.
        
        Args:
            text: Text to process
            
        Returns:
            Text with spaces inserted
        """
        # Clean text
        text = re.sub(r'[^A-Z]', '', text.upper())
        
        # Find words in text
        result = ""
        i = 0
        
        while i < len(text):
            # Try to find the longest valid word starting at position i
            found_word = False
            
            # Try different word lengths, starting with longer words
            for length in range(min(15, len(text) - i), 0, -1):
                word = text[i:i+length]
                
                # Check if it's a valid word in our dictionary
                if word in self.dictionary:
                    result += word + " "
                    i += length
                    found_word = True
                    break
            
            # If no valid word found, add a single character
            if not found_word:
                # Check if it's a valid single letter word (A, E, O)
                if len(text) > i and text[i] in self.valid_single_letters:
                    result += text[i] + " "
                else:
                    # For other single letters, just add them
                    result += text[i] + " "
                i += 1
        
        return result.strip()

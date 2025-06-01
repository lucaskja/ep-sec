#!/usr/bin/env python3
"""
Portuguese language statistics for cryptanalysis.
Based on frequency data from Brazilian Portuguese texts.
"""

# Letter frequencies in Portuguese
LETTER_FREQUENCIES = {
    'A': 14.63, 'B': 1.04, 'C': 3.88, 'D': 4.99, 'E': 12.57, 'F': 1.02,
    'G': 1.30, 'H': 1.28, 'I': 6.18, 'J': 0.40, 'K': 0.02, 'L': 2.78,
    'M': 4.74, 'N': 5.05, 'O': 10.73, 'P': 2.52, 'Q': 1.20, 'R': 6.53,
    'S': 7.81, 'T': 4.34, 'U': 4.63, 'V': 1.67, 'W': 0.01, 'X': 0.21,
    'Y': 0.01, 'Z': 0.47
}

# The 20 most frequent digrams in Portuguese (per 100 letters)
DIGRAM_FREQUENCIES = {
    'DE': 1.76, 'RA': 1.67, 'ES': 1.65, 'OS': 1.51, 'AS': 1.49,
    'DO': 1.41, 'AR': 1.33, 'CO': 1.31, 'EN': 1.23, 'QU': 1.20,
    'ER': 1.18, 'DA': 1.17, 'RE': 1.14, 'CA': 1.11, 'TA': 1.10,
    'SE': 1.08, 'NT': 1.08, 'MA': 1.06, 'UE': 1.05, 'TE': 1.05
}

# The 20 most frequent trigrams in Portuguese (per 100 letters)
TRIGRAM_FREQUENCIES = {
    'QUE': 0.96, 'ENT': 0.56, 'COM': 0.47, 'NTE': 0.44, 'EST': 0.34,
    'AVA': 0.34, 'ARA': 0.33, 'ADO': 0.33, 'PAR': 0.30, 'NDO': 0.30,
    'NAO': 0.30, 'ERA': 0.30, 'AND': 0.30, 'UMA': 0.28, 'STA': 0.28,
    'RES': 0.27, 'MEN': 0.27, 'CON': 0.27, 'DOS': 0.25, 'ANT': 0.25
}

# Common short words in Portuguese (per 100 letters)
ONE_LETTER_WORDS = {
    'E': 0.88, 'A': 0.84, 'O': 0.71
}

TWO_LETTER_WORDS = {
    'DE': 0.82, 'UM': 0.31, 'SE': 0.30, 'DA': 0.27, 'OS': 0.25,
    'DO': 0.25, 'AS': 0.19, 'EM': 0.17, 'NO': 0.14, 'NA': 0.12
}

THREE_LETTER_WORDS = {
    'QUE': 0.63, 'NAO': 0.29, 'UMA': 0.21, 'COM': 0.21, 'ERA': 0.14,
    'POR': 0.12, 'MAS': 0.11, 'DOS': 0.11, 'LHE': 0.09, 'FOI': 0.07,
    'ELE': 0.07, 'DAS': 0.07, 'SUA': 0.06, 'SEU': 0.06, 'SEM': 0.05
}

# Letter probabilities at the beginning of words
INITIAL_LETTER_FREQUENCIES = {
    'D': 12, 'A': 11, 'E': 11, 'C': 8, 'P': 7, 'S': 6, 'O': 6, 'M': 6,
    'N': 5, 'Q': 4, 'T': 4, 'F': 3, 'U': 3, 'V': 3, 'L': 2, 'R': 2,
    'B': 2, 'I': 2, 'G': 2, 'J': 1, 'H': 1, 'Z': 0, 'K': 0, 'X': 0,
    'W': 0, 'Y': 0
}

# Letter probabilities at the end of words
FINAL_LETTER_FREQUENCIES = {
    'A': 70, 'O': 65, 'E': 60, 'S': 48, 'M': 21, 'R': 14, 'U': 10,
    'I': 5, 'L': 4, 'Z': 2, 'D': 0, 'T': 0, 'H': 0, 'N': 0, 'C': 0,
    'Y': 0, 'B': 0, 'X': 0, 'V': 0, 'K': 0, 'G': 0, 'F': 0, 'P': 0,
    'W': 0, 'Q': 0, 'J': 0
}

# Other statistics
AVERAGE_WORD_LENGTH = 4.53
VOWEL_RATIO = 4.88 / 10  # Average number of vowels per 10 letters

def score_text_by_frequency(text):
    """
    Score text based on letter frequencies in Portuguese.
    
    Args:
        text: Text to score
        
    Returns:
        Score between 0 and 1, higher is better
    """
    # Count letter frequencies in the text
    letter_count = {}
    total_letters = 0
    
    for char in text.upper():
        if 'A' <= char <= 'Z':
            letter_count[char] = letter_count.get(char, 0) + 1
            total_letters += 1
    
    if total_letters == 0:
        return 0
    
    # Calculate frequency difference from expected
    score = 0
    for char, expected_freq in LETTER_FREQUENCIES.items():
        actual_freq = (letter_count.get(char, 0) / total_letters) * 100
        # Lower difference is better
        difference = abs(actual_freq - expected_freq)
        # Convert to a score where lower difference = higher score
        char_score = max(0, 1 - (difference / expected_freq))
        score += char_score * (expected_freq / 100)  # Weight by expected frequency
    
    # Normalize score
    return score / sum(LETTER_FREQUENCIES.values()) * 100

def score_digrams(text):
    """
    Score text based on digram frequencies in Portuguese.
    
    Args:
        text: Text to score
        
    Returns:
        Score between 0 and 1, higher is better
    """
    text = text.upper()
    score = 0
    
    # Count digrams in text
    digram_count = {}
    total_digrams = 0
    
    for i in range(len(text) - 1):
        digram = text[i:i+2]
        if all('A' <= c <= 'Z' for c in digram):
            digram_count[digram] = digram_count.get(digram, 0) + 1
            total_digrams += 1
    
    if total_digrams == 0:
        return 0
    
    # Score based on common digrams
    for digram, expected_freq in DIGRAM_FREQUENCIES.items():
        if digram in digram_count:
            # Each occurrence of a common digram adds to the score
            score += digram_count[digram] * expected_freq
    
    # Normalize by text length
    return score / (len(text) / 100)

def score_trigrams(text):
    """
    Score text based on trigram frequencies in Portuguese.
    
    Args:
        text: Text to score
        
    Returns:
        Score between 0 and 1, higher is better
    """
    text = text.upper()
    score = 0
    
    # Count trigrams in text
    trigram_count = {}
    total_trigrams = 0
    
    for i in range(len(text) - 2):
        trigram = text[i:i+3]
        if all('A' <= c <= 'Z' for c in trigram):
            trigram_count[trigram] = trigram_count.get(trigram, 0) + 1
            total_trigrams += 1
    
    if total_trigrams == 0:
        return 0
    
    # Score based on common trigrams
    for trigram, expected_freq in TRIGRAM_FREQUENCIES.items():
        if trigram in trigram_count:
            # Each occurrence of a common trigram adds to the score
            score += trigram_count[trigram] * expected_freq * 2  # Higher weight for trigrams
    
    # Normalize by text length
    return score / (len(text) / 100)

def score_short_words(text):
    """
    Score text based on common short words in Portuguese.
    
    Args:
        text: Text to score
        
    Returns:
        Score between 0 and 1, higher is better
    """
    # Split text into words
    words = text.upper().split()
    if not words:
        return 0
    
    score = 0
    
    # Score one-letter words
    for word in words:
        if len(word) == 1 and word in ONE_LETTER_WORDS:
            score += ONE_LETTER_WORDS[word] * 5  # Higher weight for one-letter words
    
    # Score two-letter words
    for word in words:
        if len(word) == 2 and word in TWO_LETTER_WORDS:
            score += TWO_LETTER_WORDS[word] * 3  # Higher weight for two-letter words
    
    # Score three-letter words
    for word in words:
        if len(word) == 3 and word in THREE_LETTER_WORDS:
            score += THREE_LETTER_WORDS[word] * 2  # Higher weight for three-letter words
    
    # Normalize by number of words
    return score / len(words)

def score_word_boundaries(text):
    """
    Score text based on initial and final letter frequencies in Portuguese.
    
    Args:
        text: Text to score
        
    Returns:
        Score between 0 and 1, higher is better
    """
    # Split text into words
    words = text.upper().split()
    if not words:
        return 0
    
    score = 0
    
    # Score initial letters
    for word in words:
        if word and 'A' <= word[0] <= 'Z':
            score += INITIAL_LETTER_FREQUENCIES.get(word[0], 0) / 12  # Normalize by max frequency
    
    # Score final letters
    for word in words:
        if word and 'A' <= word[-1] <= 'Z':
            score += FINAL_LETTER_FREQUENCIES.get(word[-1], 0) / 70  # Normalize by max frequency
    
    # Normalize by number of words
    return score / (len(words) * 2)  # Divide by 2 because we check both start and end

def score_vowel_ratio(text):
    """
    Score text based on vowel ratio in Portuguese.
    
    Args:
        text: Text to score
        
    Returns:
        Score between 0 and 1, higher is better
    """
    vowels = 'AEIOU'
    vowel_count = sum(1 for c in text.upper() if c in vowels)
    total_letters = sum(1 for c in text.upper() if 'A' <= c <= 'Z')
    
    if total_letters == 0:
        return 0
    
    actual_ratio = vowel_count / total_letters
    # Score based on how close the ratio is to the expected ratio
    return 1 - abs(actual_ratio - VOWEL_RATIO) / VOWEL_RATIO

def score_portuguese_text(text):
    """
    Score text based on Portuguese language statistics.
    
    Args:
        text: Text to score
        
    Returns:
        Score between 0 and 100, higher is better
    """
    # Calculate individual scores
    letter_score = score_text_by_frequency(text)
    digram_score = score_digrams(text)
    trigram_score = score_trigrams(text)
    short_word_score = score_short_words(text)
    word_boundary_score = score_word_boundaries(text)
    vowel_ratio_score = score_vowel_ratio(text)
    
    # Combine scores with weights
    total_score = (
        letter_score * 0.3 +
        digram_score * 0.25 +
        trigram_score * 0.25 +
        short_word_score * 0.1 +
        word_boundary_score * 0.05 +
        vowel_ratio_score * 0.05
    )
    
    return total_score

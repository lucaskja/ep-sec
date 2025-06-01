def score_decryption(decrypted_text: str, normalized_text: str) -> Tuple[float, bool]:
    """
    Score the decryption based on n-gram frequencies and substring matching.
    
    Args:
        decrypted_text: Decrypted text to score
        normalized_text: Normalized text for substring matching
        
    Returns:
        Tuple of (score, is_valid_match)
    """
    score = 0
    is_valid_match = False
    
    # Check for substrings in normalized text - this is the primary validation
    min_match_length = 15  # Minimum length for a valid substring match
    
    # Try to find increasingly longer matches
    for length in range(min_match_length, min(100, len(decrypted_text)), 5):
        # Check multiple positions in the decrypted text
        for start_pos in range(0, min(200, len(decrypted_text) - length), 20):  # Increased step to reduce output
            substring = decrypted_text[start_pos:start_pos+length]
            if substring in normalized_text:
                print(f"Found matching substring in normalized text: '{substring}'")
                score += length * 5
                is_valid_match = True
                
                # Try to extend the match
                extended_length = length
                while start_pos + extended_length < len(decrypted_text) and extended_length < 200:
                    extended_length += 5
                    extended_substring = decrypted_text[start_pos:start_pos+extended_length]
                    if extended_substring in normalized_text:
                        print(f"Extended match to length {extended_length}")
                        score += 25  # Bonus for longer matches
                    else:
                        break
                
                # If we found a substantial match, we can return early
                if length >= 30:
                    return score, True
    
    # If we didn't find a direct substring match, check for common Portuguese words
    if not is_valid_match:
        common_words = ["QUE", "PARA", "COM", "UMA", "ELA", "ERA", "MINHA", "MAS", "POR", "MAIS",
                       "SUA", "QUANDO", "PORQUE", "TINHA", "ESTAVA", "ELE", "DISSE", "COMO", "FOI"]
        
        word_count = 0
        for word in common_words:
            count = decrypted_text.count(word)
            if count > 0:
                word_count += count
                score += count * len(word)
        
        # If we find many common words, it might still be a valid match
        if word_count >= 5:
            print(f"Found {word_count} common Portuguese words")
            is_valid_match = True
    
    # Check letter frequencies from the data file
    letter_freqs = load_ngram_frequencies(1)
    if letter_freqs:
        # Sample the text to save computation
        sample_size = min(1000, len(decrypted_text))
        sample_step = max(1, len(decrypted_text) // sample_size)
        
        letter_counts = Counter(decrypted_text[::sample_step])
        total_letters = sum(letter_counts.values())  # Fixed: use sum of values, not len
        
        freq_score = 0
        for letter, count in letter_counts.items():
            observed_freq = count / total_letters
            expected_freq = letter_freqs.get(letter, 0)
            
            # Score based on how close the observed frequency is to the expected frequency
            similarity = 1 - min(abs(observed_freq - expected_freq) / max(expected_freq, 0.001), 1)
            freq_score += similarity * 10
        
        # Normalize the frequency score
        freq_score = freq_score / len(letter_counts) if letter_counts else 0
        score += freq_score
    
    # Check vowel ratio (Portuguese has ~46% vowels)
    vowels = sum(1 for c in decrypted_text[:1000] if c in 'AEIOU')  # Only check first 1000 chars
    vowel_ratio = vowels / min(1000, len(decrypted_text)) if decrypted_text else 0
    
    # Portuguese texts typically have 40-50% vowels
    if 0.4 <= vowel_ratio <= 0.5:
        score += 50
    elif 0.35 <= vowel_ratio <= 0.55:
        score += 25
    
    return score, is_valid_match

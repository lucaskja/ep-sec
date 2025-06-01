#!/usr/bin/env python3
"""
Script to update the advanced frequency analyzer with the extracted n-gram frequencies.
"""

import os
import re

def main():
    """Main function."""
    # Check if portuguese_ngrams.py exists
    if not os.path.exists("src/portuguese_ngrams.py"):
        print("Error: src/portuguese_ngrams.py not found. Run extract_ngram_frequencies.py first.")
        return
    
    # Check if advanced_frequency_analyzer.py exists
    if not os.path.exists("src/advanced_frequency_analyzer.py"):
        print("Error: src/advanced_frequency_analyzer.py not found.")
        return
    
    # Read the advanced_frequency_analyzer.py file
    with open("src/advanced_frequency_analyzer.py", 'r') as f:
        content = f.read()
    
    # Add import for portuguese_ngrams
    if "from src.portuguese_ngrams import" not in content and "from portuguese_ngrams import" not in content:
        # Find the last import statement
        import_section = re.search(r'(import.*?)(?=\n\n)', content, re.DOTALL)
        if import_section:
            # Add import after the last import statement
            new_import = import_section.group(1) + "\n\n# Import Portuguese n-grams\ntry:\n    from src.portuguese_ngrams import COMMON_NGRAMS, LETTER_FREQUENCIES, DIGRAMS, TRIGRAMS, QUADGRAMS, PENTAGRAMS\nexcept ImportError:\n    try:\n        from portuguese_ngrams import COMMON_NGRAMS, LETTER_FREQUENCIES, DIGRAMS, TRIGRAMS, QUADGRAMS, PENTAGRAMS\n    except ImportError:\n        print(\"Warning: portuguese_ngrams.py not found. Using default n-grams.\")\n"
            content = content.replace(import_section.group(1), new_import)
    
    # Update the setup_common_ngrams method
    setup_method = re.search(r'def setup_common_ngrams\(self\):(.*?)(?=\n    def|\n\n)', content, re.DOTALL)
    if setup_method:
        # Create new setup_common_ngrams method
        new_setup = """def setup_common_ngrams(self):
        \"\"\"Set up common n-grams for Portuguese language based on matrix size.\"\"\"
        # Use extracted n-grams if available
        if 'COMMON_NGRAMS' in globals():
            self.common_ngrams = COMMON_NGRAMS
        else:
            # Fallback to default n-grams
            self.common_ngrams = {
                3: ["QUE", "ENT", "COM", "ROS", "IST", "ADO", 
                    "ELA", "PRA", "INH", "EST", "NTE", "ERA", "AND", "UMA", "STA", 
                    "RES", "MEN", "CON", "DOS", "ANT"],
                4: ["VOCE", "INHA", "PARA", "AQUE", "EVOC", "ANDO", "OQUE", "ESTA", 
                    "TAVA", "ENTE", "EQUE", "RQUE", "MINH", "OCES", "ENAO", "ENTA", 
                    "MENT", "QUEE", "STAV", "NHAM"],
                5: ["EVOCE", "MINHA", "VOCES", "STAVA", "INHAM", "ESTAV", "OVOCE", 
                    "ORQUE", "TINHA", "NHAMA", "PORQU", "HAMAE", "AQUEL", "UEVOC", 
                    "QUEVO", "UANDO", "QUAND", "AVOCE", "DISSE", "EPOIS"]
            }
        
        # Common Portuguese words for validation
        self.common_words = ["DE", "QUE", "E", "A", "O", "DA", "DO", "EM", "PARA", "COM",
                            "NAO", "UMA", "OS", "NO", "SE", "NA", "POR", "MAIS", "AS", "DOS"]"""
        
        content = content.replace(setup_method.group(0), new_setup)
    
    # Update the score_decryption method to use LETTER_FREQUENCIES if available
    score_method = re.search(r'def score_decryption\(self, decrypted_text: str\) -> float:(.*?)(?=\n    def|\n\n)', content, re.DOTALL)
    if score_method:
        # Find the letter frequencies part
        letter_freq_part = re.search(r'expected_freq = ([^,]+)', score_method.group(1))
        if letter_freq_part:
            # Replace with the extracted letter frequencies
            new_letter_freq = "expected_freq = LETTER_FREQUENCIES.get(letter, 0) / 100 if 'LETTER_FREQUENCIES' in globals() else 0.001"
            content = content.replace(letter_freq_part.group(0), new_letter_freq)
    
    # Write the updated content back to the file
    with open("src/advanced_frequency_analyzer.py", 'w') as f:
        f.write(content)
    
    print("Updated advanced_frequency_analyzer.py with extracted n-gram frequencies.")

if __name__ == "__main__":
    main()

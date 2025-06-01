#!/usr/bin/env python3
"""
Configuration settings for Hill Cipher Breaker.
"""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DICT_PATH = os.path.join(PROJECT_ROOT, "data", "portuguese_dict.txt")
DICT_URL = "https://www.ime.usp.br/~pf/dicios/br-utf8.txt"

# Directories
REPORTS_DIR = os.path.join(PROJECT_ROOT, "relatorios")
KNOWN_TEXTS_DIR = os.path.join(PROJECT_ROOT, "textos_conhecidos")
UNKNOWN_TEXTS_DIR = os.path.join(PROJECT_ROOT, "textos_desconhecidos")

# Create directories if they don't exist
os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Language model parameters
COMMON_BIGRAMS = [
    "DE", "RA", "ES", "OS", "AR", "QU", "NT", "EN", "ER", "RE",
    "TE", "CO", "OR", "AS", "DO", "AD", "TA", "SE", "ME", "AN"
]

COMMON_TRIGRAMS = [
    "QUE", "EST", "COM", "NTE", "TEM", "ARA", "POR", "ENT", "TER", "CON",
    "RES", "ADE", "ERA", "ADO", "STA", "PAR", "NTO", "AND", "DES", "ESS"
]

COMMON_QUADGRAMS = [
    "MENT", "ENTE", "PARA", "ANDO", "AQUE", "ESTA", "OQUE", "COMO", "ADOS", "NTES",
    "ISTA", "IDAD", "DESS", "ANTE", "ANDO", "CONT", "ESSE", "NTOS", "PRES", "NCIA"
]

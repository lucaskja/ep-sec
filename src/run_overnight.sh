#!/bin/bash
# Script para executar o quebrador híbrido durante a noite

# Criar diretórios necessários
mkdir -p relatorios/hibrido/conhecidos
mkdir -p relatorios/hibrido/desconhecidos

# Registrar hora de início
echo "Iniciando processamento em $(date)" > relatorios/hibrido/log.txt

# Executar o quebrador híbrido
echo "Executando quebrador híbrido..."
python3 src/hill_cipher_hybrid_fixed.py

# Registrar hora de término
echo "Processamento concluído em $(date)" >> relatorios/hibrido/log.txt

# Resumo dos resultados
echo "Resumo dos resultados:" >> relatorios/hibrido/log.txt
cat relatorios/hibrido/resumo.txt >> relatorios/hibrido/log.txt

echo "Processamento concluído. Verifique os resultados em relatorios/hibrido/"

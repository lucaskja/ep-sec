#!/bin/bash
# Script para executar o quebrador híbrido durante a noite (versão corrigida)

# Criar diretórios necessários
mkdir -p relatorios/hibrido_fixed/conhecidos
mkdir -p relatorios/hibrido_fixed/desconhecidos

# Registrar hora de início
echo "Iniciando processamento em $(date)" > relatorios/hibrido_fixed/log.txt

# Executar o quebrador híbrido
echo "Executando quebrador híbrido corrigido..."
python3 src/hill_cipher_hybrid_fixed.py

# Registrar hora de término
echo "Processamento concluído em $(date)" >> relatorios/hibrido_fixed/log.txt

# Resumo dos resultados
echo "Resumo dos resultados:" >> relatorios/hibrido_fixed/log.txt
cat relatorios/hibrido/resumo.txt >> relatorios/hibrido_fixed/log.txt

echo "Processamento concluído. Verifique os resultados em relatorios/hibrido/"

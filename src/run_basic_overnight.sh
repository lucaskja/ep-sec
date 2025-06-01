#!/bin/bash
# Script para executar o quebrador básico durante a noite (apenas matrizes 2x2 e 3x3)

# Criar diretórios necessários
mkdir -p relatorios/basico_overnight/conhecidos
mkdir -p relatorios/basico_overnight/desconhecidos

# Registrar hora de início
echo "Iniciando processamento em $(date)" > relatorios/basico_overnight/log.txt

# Executar o quebrador híbrido apenas para matrizes 2x2 e 3x3
echo "Executando quebrador básico para matrizes 2x2 e 3x3..."
python3 src/hill_cipher_hybrid_fixed.py --sizes 2 3

# Registrar hora de término
echo "Processamento concluído em $(date)" >> relatorios/basico_overnight/log.txt

# Copiar resumo dos resultados
echo "Resumo dos resultados:" >> relatorios/basico_overnight/log.txt
cat relatorios/hibrido/resumo.txt >> relatorios/basico_overnight/log.txt

echo "Processamento concluído. Verifique os resultados em relatorios/hibrido/"

# Mostrar os melhores resultados
echo "Melhores resultados para matriz 2x2 (texto conhecido):"
cat relatorios/hibrido/conhecidos/hill_2x2/relatorio.txt | grep -A 3 "Resultado #2"

echo "Melhores resultados para matriz 2x2 (texto desconhecido):"
cat relatorios/hibrido/desconhecidos/hill_2x2/relatorio.txt | grep -A 3 "Resultado #2"

#!/bin/bash
# Script to run the hybrid breaker overnight

# Create necessary directories
mkdir -p relatorios/hibrido/conhecidos
mkdir -p relatorios/hibrido/desconhecidos

# Record start time
echo "Starting processing at $(date)" > relatorios/hibrido/log.txt

# Run the hybrid breaker
echo "Running hybrid breaker..."
python3 src/hill_cipher_hybrid_fixed.py

# Record end time
echo "Processing completed at $(date)" >> relatorios/hibrido/log.txt

# Results summary
echo "Results summary:" >> relatorios/hibrido/log.txt
cat relatorios/hibrido/resumo.txt >> relatorios/hibrido/log.txt

echo "Processing completed. Check results in relatorios/hibrido/"

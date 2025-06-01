# Hill Cipher Breaker

Este projeto implementa diferentes versões de quebradores para a cifra de Hill, com suporte para matrizes de tamanhos 2x2, 3x3, 4x4 e 5x5.

## Estrutura do Projeto

```
.
├── src/                            # Códigos fonte
│   ├── utils.py                    # Funções utilitárias
│   ├── config.py                   # Configurações do projeto
│   ├── hill_cipher.py              # Implementação da cifra de Hill
│   ├── hill_cipher_breaker.py      # Versão básica
│   ├── hill_cipher_breaker_advanced.py  # Versão avançada
│   ├── hill_cipher_breaker_optimized.py # Versão otimizada
│   ├── hill_cipher_hybrid.py       # Versão híbrida
│   ├── enhanced_hill_breaker.py    # Versão aprimorada
│   ├── portuguese_statistics.py    # Estatísticas da língua portuguesa
│   └── test_scripts/               # Scripts de teste
│
├── data/                           # Dados auxiliares
│   └── portuguese_dict.txt         # Dicionário de português
│
├── relatorios/                     # Relatórios organizados
│   ├── basico/                     # Relatórios da versão básica
│   ├── avancado/                   # Relatórios da versão avançada
│   ├── otimizado/                  # Relatórios da versão otimizada
│   ├── hibrido/                    # Relatórios da versão híbrida
│   └── enhanced/                   # Relatórios da versão aprimorada
│
├── textos_conhecidos/              # Textos conhecidos para teste
│   ├── Aberto/                     # Textos abertos
│   ├── Cifrado/                    # Textos cifrados
│   └── textos/                     # Textos originais
│
└── textos_desconhecidos/           # Textos desconhecidos para teste
    ├── Aberto/                     # Textos abertos
    ├── Cifrado/                    # Textos cifrados
    └── textos/                     # Textos originais
```

## Versões do Quebrador

### Versão Básica (`hill_cipher_breaker.py`)
- Implementação inicial com suporte para matrizes 2x2 e 3x3
- Força bruta simples
- Análise estatística básica

### Versão Avançada (`hill_cipher_breaker_advanced.py`)
- Suporte para matrizes 2x2, 3x3, 4x4 e 5x5
- Processamento paralelo
- Dicionário de palavras em português
- Técnicas de redução do espaço de busca

### Versão Otimizada (`hill_cipher_breaker_optimized.py`)
- Melhorias na análise de palavras
- Segmentação de texto sem espaços
- Ataque com texto conhecido aprimorado
- Sistema de pontuação sofisticado
- Pós-processamento do texto decifrado

### Versão Híbrida (`hill_cipher_hybrid.py`)
- Combina o quebrador básico para matrizes 2x2
- Usa o quebrador otimizado para matrizes 3x3, 4x4 e 5x5
- Melhor equilíbrio entre velocidade e precisão

### Versão Aprimorada (`enhanced_hill_breaker.py`)
- Análise estatística baseada em frequências da língua portuguesa
- Processamento extremamente rápido sem dependência de dicionário
- Priorização de matrizes conhecidas por funcionarem bem
- Sistema de pontuação baseado em n-gramas (bigramas e trigramas)
- Paralelização otimizada com balanceamento dinâmico de carga

## Como Executar

### Versão Básica
```bash
python3 src/hill_cipher_breaker.py
```

### Versão Avançada
```bash
python3 src/hill_cipher_breaker_advanced.py
```

### Versão Otimizada
```bash
python3 src/hill_cipher_breaker_optimized.py
```

### Versão Híbrida
```bash
python3 src/hill_cipher_hybrid.py
```

### Versão Aprimorada
```bash
python3 src/run_enhanced.py
```

### Scripts para Windows
```powershell
.\src\run_overnight.ps1  # Execução noturna da versão híbrida
.\src\run_enhanced.ps1   # Execução da versão aprimorada
```

## Técnicas Implementadas

1. **Ataque com Texto Conhecido**
   - Uso de pares de texto claro e cifrado
   - Resolução de sistemas de equações lineares

2. **Análise Estatística**
   - Frequência de letras em português
   - Análise de n-gramas (bigramas, trigramas, quadrigramas)
   - Pontuação baseada em estatísticas linguísticas

3. **Força Bruta Otimizada**
   - Processamento paralelo
   - Cache de matrizes inversíveis
   - Geração inteligente de candidatos
   - Balanceamento dinâmico de carga

4. **Redução do Espaço de Busca**
   - Filtro por determinante
   - Uso de estruturas de blocos para matrizes maiores
   - Heurística de janela (Shutter)
   - Priorização de matrizes conhecidas

5. **Segmentação de Texto**
   - Identificação de palavras em texto sem espaços
   - Programação dinâmica para encontrar a melhor segmentação

6. **Pós-processamento**
   - Adição de espaços e pontuação
   - Formatação do texto para melhor legibilidade

## Requisitos

- Python 3.6+
- NumPy
- Acesso à internet para baixar o dicionário de português (opcional)

## Melhorias Recentes

Recentemente, implementamos várias melhorias significativas no quebrador de cifra de Hill:

### 1. Análise Estatística Pura
- Substituição do dicionário por análise estatística baseada em frequências
- Implementação de pontuação baseada em bigramas e trigramas
- Análise de palavras curtas comuns em português
- Verificação de proporção de vogais e consoantes

### 2. Otimização de Desempenho
- Redução drástica do tempo de processamento (de minutos para segundos)
- Paralelização com balanceamento dinâmico de carga
- Parada antecipada quando boas soluções são encontradas
- Priorização de matrizes conhecidas por funcionarem bem

### 3. Melhor Detecção de Soluções
- Pontuação mais precisa para identificar textos em português
- Detecção de palavras comuns como indicador de solução correta
- Bônus para matrizes conhecidas por funcionarem bem
- Sistema de pontuação ponderado por frequências reais

## Resultados Comparativos

| Versão | Tempo (2x2) | Tempo (3x3) | Precisão |
|--------|------------|------------|----------|
| Básica | <1s | ~60s | Baixa |
| Avançada | ~20s | ~300s | Média |
| Otimizada | ~400s | ~1800s | Alta |
| Híbrida | ~30s | ~1200s | Alta |
| Aprimorada | ~5s | ~15s | Alta |

A versão aprimorada oferece o melhor equilíbrio entre velocidade e precisão, sendo capaz de quebrar cifras de Hill rapidamente com alta taxa de sucesso.

## Estratégias para Redução do Espaço de Busca

Para matrizes maiores (3x3, 4x4, 5x5), implementamos estratégias específicas para reduzir o enorme espaço de busca:

1. **Análise Estatística**
   - Uso de frequências de letras, bigramas e trigramas em português
   - Pontuação baseada em estatísticas linguísticas reais
   - Detecção de padrões comuns da língua portuguesa

2. **Estruturas Específicas**
   - Matrizes triangulares com elementos diagonais coprimos
   - Matrizes com estrutura de blocos
   - Matrizes com padrões específicos (diagonal dominante, circulante, simétrica)

3. **Paralelização Inteligente**
   - Balanceamento dinâmico de carga entre threads
   - Processamento prioritário de matrizes promissoras
   - Parada antecipada quando boas soluções são encontradas

Estas estratégias reduzem o espaço de busca em várias ordens de magnitude, tornando viável a quebra da cifra de Hill com matrizes grandes em tempo razoável.

## Autores

- Lucas Kledeglau Jahchan Alves
- Universidade de São Paulo (USP)
- Disciplina de Segurança de Dados

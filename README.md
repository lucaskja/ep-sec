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
│   └── test_scripts/               # Scripts de teste
│
├── data/                           # Dados auxiliares
│   └── portuguese_dict.txt         # Dicionário de português
│
├── relatorios/                     # Relatórios organizados
│   ├── basico/                     # Relatórios da versão básica
│   ├── avancado/                   # Relatórios da versão avançada
│   ├── otimizado/                  # Relatórios da versão otimizada
│   └── hibrido/                    # Relatórios da versão híbrida
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

### Scripts de Teste
```bash
python3 src/test_hybrid.py
python3 src/compare_breakers.py
```

## Técnicas Implementadas

1. **Ataque com Texto Conhecido**
   - Uso de pares de texto claro e cifrado
   - Resolução de sistemas de equações lineares

2. **Análise Estatística**
   - Frequência de letras em português
   - Análise de n-gramas (bigramas, trigramas, quadrigramas)

3. **Força Bruta Otimizada**
   - Processamento paralelo
   - Cache de matrizes inversíveis
   - Geração inteligente de candidatos

4. **Redução do Espaço de Busca**
   - Filtro por determinante
   - Uso de estruturas de blocos para matrizes maiores
   - Heurística de janela (Shutter)

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

Recentemente, implementamos várias melhorias significativas no quebrador de cifra de Hill otimizado:

### 1. Análise de Similaridade Aprimorada
- Uso de múltiplos tamanhos de n-gramas (3, 4, 5 e 6) para análise mais completa
- Pesos diferentes para n-gramas de diferentes tamanhos
- Comparação direta com textos conhecidos para validação de resultados

### 2. Segmentação de Texto Avançada
- Reconhecimento de prefixos e sufixos de palavras
- Identificação de pares comuns de palavras
- Algoritmo de programação dinâmica otimizado para segmentação

### 3. Sistema de Feedback Inteligente
- Feedback em tempo real durante o processamento
- Exibição do melhor resultado atual durante as iterações
- Uso de resultados de similaridade para guiar a busca

### 4. Otimizações de Desempenho
- Cache para evitar processamento duplicado
- Paralelização mais eficiente com ThreadPoolExecutor
- Sistema de cache para resultados de similaridade

Estas melhorias resultaram em um aumento significativo na qualidade dos resultados, com até 46.67% de palavras válidas identificadas e 29.97% de similaridade com textos conhecidos.

## Resultados Comparativos

| Versão | Palavras Válidas | Similaridade | Tempo (2x2) |
|--------|-----------------|-------------|-------------|
| Básica | ~20% | N/A | <1s |
| Avançada | ~30% | ~10% | ~20s |
| Otimizada | ~46% | ~30% | ~400s |

A versão otimizada, embora mais lenta, produz resultados significativamente melhores, especialmente para matrizes de tamanhos maiores.

## Estratégias para Redução do Espaço de Busca em Matrizes 3x3

Para matrizes 3x3, implementamos estratégias específicas para reduzir o enorme espaço de busca (26^9):

1. **Filtro por Determinante Válido**
   - Apenas matrizes com determinante coprimo com 26 são consideradas

2. **Estruturas Específicas**
   - Matrizes triangulares com elementos diagonais coprimos
   - Matrizes com estrutura de blocos
   - Matrizes com padrões específicos (diagonal dominante, circulante, simétrica)

3. **Análise de Frequência**
   - Priorização de matrizes com elementos baseados em frequências de letras

4. **Amostragem Inteligente**
   - Elementos diagonais com maior probabilidade de serem coprimos com 26

Estas estratégias reduzem o espaço de busca em mais de 11 ordens de magnitude, tornando viável a quebra da cifra de Hill com matrizes 3x3 em tempo razoável.

## Autores

- Lucas Kledeglau Jahchan Alves
- Universidade de São Paulo (USP)
- Disciplina de Segurança de Dados

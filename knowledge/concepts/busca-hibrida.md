---
type: Concept
title: Busca lexical, busca vetorial e fusão RRF
description: Papel do índice invertido BM25 e do k-NN semântico.
tags: [opensearch, bm25, knn, vetor, pesquisa, busca, indice]
status: draft
---

# Busca híbrida

- **BM25** privilegia coincidências textuais, especialmente marca, modelo e versão.
- **k-NN** aproxima intenção de uso e descrição semântica, usando embeddings do Ollama.
- **RRF** combina a posição dos documentos em rankings independentes, sem somar diretamente scores BM25 e vetoriais.
- **Filtros estruturados** de preço, ano, carroceria e câmbio são aplicados em ambos os mecanismos.

O OpenSearch retorna candidatos; o LLM não conhece os registros que não foram recuperados.

---
type: Architecture Decision
title: Filtragem determinística antes da geração
status: draft
description: Regras executáveis de preço e características não dependem do prompt do modelo.
tags: [arquitetura, decisao, filtros, seguranca, negocio]
---

# Decisão

Usar `src/catalog.py` para extrair apenas um conjunto explícito de condições conhecidas e enviar `term`/`range` ao OpenSearch. A recuperação híbrida também deve receber os filtros.

# Consequências

- A mesma condição de preço vale para busca BM25 e k-NN.
- Intenções não compreendidas por regex podem ficar sem filtro: documentar limitação e adicionar casos de teste.
- Para produção, coletar filtros estruturados da interface ou usar parser estruturado com validação forte, sem deixar o modelo construir consultas arbitrárias.

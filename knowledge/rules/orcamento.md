---
type: Business Rule
title: Limites de orçamento em consultas de veículos
description: Preço máximo e mínimo devem ser filtros reais aplicados ao catálogo.
tags: [preco, orcamento, filtros, veiculos]
status: draft
---

# Limites de orçamento

- Expressões como "até R$ 90 mil" representam um limite **inclusivo** (preço <= 90000).
- O OpenSearch aplica o filtro numérico no campo `preco` antes de retornar candidatos.
- Nunca aceite uma comparação inventada pelo modelo como evidência de preço.
- Valores do catálogo são **fictícios**. Não se trata de oferta, cotação ou disponibilidade atual.
- Se a pergunta estiver fora das expressões suportadas pelo parser, o sistema pode não detectar o preço: validar antes de decisões reais.

Consulte [filtros determinísticos](/decisions/filtros-deterministicos.md).

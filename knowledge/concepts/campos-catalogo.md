---
type: Data Contract
title: Contrato do catálogo fictício
resource: /data/vehicles.json
description: Campos disponíveis para recuperação e comparação no catálogo didático.
tags: [schema, dados, veiculos, catalogo, detalhes]
status: draft
---

# Campos do catálogo

Campos: `id`, `marca`, `modelo`, `versao`, `ano`, `preco`, `km`, `carroceria`, `cambio`, `combustivel`, `destaques`, `descricao`.

`texto_busca` é campo técnico derivado. `vetor` é uma representação numérica gerada na indexação e não deve aparecer em respostas da API.

**Não existem neste dataset** valores em litros para porta-malas, consumo homologado, preço de mercado, fotos, localização, disponibilidade, financiamento ou resultado de crash-test.

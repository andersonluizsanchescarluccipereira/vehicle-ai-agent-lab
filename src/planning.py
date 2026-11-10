"""Planejamento determinístico de ferramentas; sem dependência de LLM."""
from typing import Any
from .catalog import normalize


def requires_compare(question: str) -> bool:
    q = normalize(question)
    return any(word in q for word in ('compar', 'diferenc', 'versus', ' vs ', 'melhor entre'))


def plan(question: str) -> dict[str, Any]:
    compare = requires_compare(question)
    return {'comparar': compare, 'plano': ['buscar_conhecimento', 'buscar_veiculos'] +
            (['comparar_veiculos'] if compare else []) + ['responder']}

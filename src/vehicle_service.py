"""Casos de uso de domínio: sem SDK MCP e sem regras de autorização pelo modelo."""
from __future__ import annotations
from typing import Any

from . import config
from .catalog import extract_filters
from .fusion import rrf_merge

# Apenas campos de catálogo permitidos nas respostas. Nunca retorne vetor ao LLM.
FIELDS = ('id', 'marca', 'modelo', 'versao', 'ano', 'preco', 'km', 'carroceria',
          'cambio', 'combustivel', 'destaques', 'descricao', 'rrf_score')


def public_vehicle(vehicle: dict[str, Any]) -> dict[str, Any]:
    return {key: vehicle[key] for key in FIELDS if key in vehicle}


def matches_filters(vehicle: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Validação de defesa em profundidade após OpenSearch (não confiar só no LLM)."""
    for key in ('carroceria', 'cambio', 'combustivel', 'marca'):
        if key in filters and str(vehicle.get(key, '')).lower() != filters[key]:
            return False
    checks = [('preco_max', 'preco', lambda n, limit: n <= limit),
              ('preco_min', 'preco', lambda n, limit: n >= limit),
              ('km_max', 'km', lambda n, limit: n <= limit),
              ('ano_min', 'ano', lambda n, limit: n >= limit)]
    for criterion, field, comparator in checks:
        if criterion in filters:
            if field not in vehicle or not isinstance(vehicle[field], (int, float)):
                return False
            if not comparator(vehicle[field], filters[criterion]):
                return False
    return True


class VehicleService:
    def __init__(self, search: Any, client: Any):
        self.search = search
        self.client = client

    def search_vehicles(self, question: str, mode: str = 'hybrid', limit: int = 5) -> dict[str, Any]:
        if not 1 <= len(question.strip()) <= 500:
            raise ValueError('A pergunta deve ter entre 1 e 500 caracteres.')
        if mode not in {'hybrid', 'bm25', 'vector'}:
            raise ValueError('Modo inválido.')
        limit = max(1, min(int(limit), 5))
        filters = extract_filters(question)
        size = max(config.TOP_K, limit)
        lexical = self.search.bm25(question, filters, size) if mode != 'vector' else []
        semantic = self.search.vector(question, filters, size) if mode != 'bm25' else []
        combined = rrf_merge(lexical, semantic, size=size)
        results = [public_vehicle(v) for v in combined if matches_filters(v, filters)][:limit]
        return {'mode': mode, 'filters': filters, 'vehicles': results,
                'sources': [f"catalog:{v['id']}" for v in results]}

    def details(self, vehicle_id: str) -> dict[str, Any]:
        import re
        if not re.fullmatch(r'V\d{3}', vehicle_id):
            raise ValueError('ID precisa seguir padrão V001.')
        try:
            result = self.client.get(index=config.OPENSEARCH_INDEX, id=vehicle_id,
                                     _source_excludes=['vetor'])
        except Exception as exc:
            from opensearchpy.exceptions import NotFoundError
            if isinstance(exc, NotFoundError):
                return {'found': False, 'vehicle_id': vehicle_id}
            raise
        return {'found': True, 'vehicle': public_vehicle({'id': vehicle_id, **result['_source']}),
                'source': f'catalog:{vehicle_id}'}

    def compare(self, vehicle_ids: list[str]) -> dict[str, Any]:
        ids = list(dict.fromkeys(vehicle_ids))[:3]
        if not 2 <= len(ids) <= 3:
            raise ValueError('Compare 2 ou 3 IDs distintos.')
        details = [self.details(v) for v in ids]
        return {'vehicles': [d['vehicle'] for d in details if d['found']],
                'not_found': [d['vehicle_id'] for d in details if not d['found']],
                'criteria': ['preco', 'ano', 'km', 'cambio', 'combustivel'],
                'sources': [f'catalog:{v}' for v in ids]}

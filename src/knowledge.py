"""Leitor OKF v0.2: arquivos Markdown tipados, sem executar conteúdo do bundle."""
from __future__ import annotations
import re
import unicodedata
from pathlib import Path
from typing import Any
import yaml
from .config import ROOT

KNOWLEDGE_ROOT = ROOT / 'knowledge'
STOPWORDS = {'quero', 'uma', 'um', 'para', 'com', 'que', 'sobre', 'por', 'até', 'ate',
             'qual', 'quais', 'me', 'os', 'as', 'de', 'do', 'da', 'e', 'ou', 'em',
             'meu', 'minha', 'veiculo', 'veiculos', 'carro', 'carros', 'comparar'}


def normalize(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', text.lower())
                   if not unicodedata.combining(c))


def tokens(text: str) -> set[str]:
    return {w for w in re.findall(r'[a-z]{3,}', normalize(text)) if w not in STOPWORDS}


def read_concept(path: Path, base: Path = KNOWLEDGE_ROOT) -> dict[str, Any]:
    """Apenas caminhos dentro da raiz permitida; frontmatter com type obrigatório."""
    resolved = path.resolve()
    resolved.relative_to(base.resolve())  # bloqueia path traversal
    raw = resolved.read_text(encoding='utf-8')
    match = re.match(r'\A---\s*\n(.*?)\n---\s*\n(.*)\Z', raw, re.DOTALL)
    if not match:
        raise ValueError(f'Frontmatter inválido: {path.name}')
    metadata = yaml.safe_load(match.group(1))
    if not isinstance(metadata, dict) or not isinstance(metadata.get('type'), str) or not metadata['type'].strip():
        raise ValueError(f'OKF exige type não vazio: {path.name}')
    return {'id': resolved.relative_to(base.resolve()).as_posix(),
            'type': metadata['type'],
            'title': str(metadata.get('title') or path.stem),
            'description': str(metadata.get('description') or ''),
            'tags': metadata.get('tags') if isinstance(metadata.get('tags'), list) else [],
            'status': str(metadata.get('status') or 'stable'),
            'verified': bool(metadata.get('verified')),
            'content': match.group(2).strip()[:5000]}


def list_concepts(base: Path = KNOWLEDGE_ROOT) -> list[dict[str, Any]]:
    concepts = []
    for path in sorted(base.rglob('*.md')):
        if path.name in {'index.md', 'log.md'}:
            continue
        concepts.append(read_concept(path, base))
    return concepts


def search_concepts(question: str, limit: int = 3, base: Path = KNOWLEDGE_ROOT) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 5))
    query_terms = tokens(question)
    results = []
    for concept in list_concepts(base):
        if concept['status'] == 'deprecated':
            continue
        title_tags = tokens(' '.join([concept['title'], concept['description'], *[str(t) for t in concept['tags']]]))
        body_terms = tokens(concept['content'])
        score = len(query_terms & title_tags) * 3 + len(query_terms & body_terms)
        if score > 0:
            results.append({**concept, 'relevance': score, 'source': f"okf:{concept['id']}"})
    return sorted(results, key=lambda d: (-d['relevance'], d['id']))[:limit]

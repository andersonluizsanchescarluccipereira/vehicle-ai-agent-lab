"""Servidor MCP: acesso controlado ao bundle OKF local."""
import json
from fastmcp import FastMCP
from src.knowledge import search_concepts, list_concepts

mcp = FastMCP('vehicle-knowledge-okf')


@mcp.tool()
def buscar_conhecimento(pergunta: str, limite: int = 3) -> str:
    """Recupera regras e conceitos do bundle OKF, indicando IDs de origem."""
    if not 1 <= len(pergunta.strip()) <= 500:
        raise ValueError('A pergunta deve ter entre 1 e 500 caracteres.')
    return json.dumps({'concepts': search_concepts(pergunta, limite)}, ensure_ascii=False)


@mcp.tool()
def listar_conceitos() -> str:
    """Lista apenas metadados e identificadores de conceitos do conhecimento OKF."""
    return json.dumps({'concepts': [
        {k: v for k, v in concept.items() if k != 'content'} for concept in list_concepts()
    ]}, ensure_ascii=False)


if __name__ == '__main__':
    mcp.run(transport='stdio')

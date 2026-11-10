"""Adaptador MCP -> LangChain BaseTool; stdio local de baixo custo."""
from __future__ import annotations
import json
import os
import sys
from typing import Any

from .config import ROOT


def build_client():
    """Cliente MCP mantido pelo LangChain >=1.4 (MCPAdapter, API beta)."""
    from langchain.mcp import MCPAdapter
    config = {
        'mcpServers': {
            'veiculos': {
                'command': sys.executable,
                'args': ['-m', 'mcp_servers.vehicles'],
                'env': {**os.environ, 'PYTHONPATH': str(ROOT)},
            },
            'conhecimento': {
                'command': sys.executable,
                'args': ['-m', 'mcp_servers.knowledge'],
                'env': {**os.environ, 'PYTHONPATH': str(ROOT)},
            },
        }
    }
    return MCPAdapter(config)


def normalize_tool_names(mcp_tools: list) -> dict[str, Any]:
    """Remove prefixo automático de servidor (`veiculos_buscar_veiculos`)."""
    known = {'buscar_veiculos', 'detalhar_veiculo', 'comparar_veiculos',
             'buscar_conhecimento', 'listar_conceitos'}
    named = {}
    for tool in mcp_tools:
        matched = [name for name in known if tool.name == name or tool.name.endswith('_' + name)]
        if len(matched) == 1:
            named[matched[0]] = tool
    return named


def parse_tool_result(result: Any) -> dict:
    """Converte retorno text/json comum dos adaptadores MCP sem assumir versão."""
    if getattr(result, 'status', None) == 'error':
        raise RuntimeError(f'Ferramenta MCP retornou erro: {str(getattr(result, "content", ""))[:200]}')
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        return json.loads(result)
    if isinstance(result, list):
        parts = []
        for part in result:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get('text'):
                parts.append(part['text'])
            elif getattr(part, 'text', None):
                parts.append(part.text)
        if parts:
            return json.loads('\n'.join(parts))
    if hasattr(result, 'content'):
        return parse_tool_result(result.content)
    raise ValueError(f'Tipo de retorno MCP inesperado: {type(result)}')


async def call_tool(tools: dict, tool_name: str, args: dict) -> dict:
    if tool_name not in tools:
        raise RuntimeError(f'Tool MCP não disponibilizada: {tool_name}')
    return parse_tool_result(await tools[tool_name].ainvoke(args))

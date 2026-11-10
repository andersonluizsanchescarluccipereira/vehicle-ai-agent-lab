"""Smoke de handshake MCP sem exigir OpenSearch/Ollama online."""
import asyncio
from src.mcp_client import build_client, normalize_tool_names


async def main():
    async with build_client() as adapter:
        names = normalize_tool_names(await adapter.list_tools())
        required = {'buscar_veiculos', 'detalhar_veiculo', 'comparar_veiculos',
                    'buscar_conhecimento', 'listar_conceitos'}
        for name in sorted(names):
            print(f'OK: {name}')
        missing = required - names.keys()
        if missing:
            raise RuntimeError(f'Tools ausentes: {missing}')
    print('Conexão MCP OK: 2 servidores, 5 tools descobertas.')


if __name__ == '__main__':
    asyncio.run(main())

"""Servidor MCP: tools de catálogo e comparação (somente leitura)."""
import json
from functools import lru_cache
from fastmcp import FastMCP
from langchain_ollama import OllamaEmbeddings
from src import config
from src.retrieval import VehicleSearch, open_client
from src.vehicle_service import VehicleService

mcp = FastMCP('vehicle-catalog')


@lru_cache(maxsize=1)
def service() -> VehicleService:
    client = open_client()
    embeddings = OllamaEmbeddings(model=config.OLLAMA_EMBED_MODEL,
                                  base_url=config.OLLAMA_BASE_URL)
    return VehicleService(VehicleSearch(client, embeddings), client)


@mcp.tool()
def buscar_veiculos(pergunta: str, modo: str = 'hybrid', limite: int = 5) -> str:
    """Busca veículos fictícios por BM25, vetor ou híbrido; aplica filtros de preço/câmbio/ano."""
    return json.dumps(service().search_vehicles(pergunta, modo, limite), ensure_ascii=False)


@mcp.tool()
def detalhar_veiculo(id_veiculo: str) -> str:
    """Consulta informações do catálogo por ID (formato V001)."""
    return json.dumps(service().details(id_veiculo), ensure_ascii=False)


@mcp.tool()
def comparar_veiculos(ids: list[str]) -> str:
    """Compara de 2 a 3 veículos existentes por atributos objetivos do catálogo."""
    return json.dumps(service().compare(ids), ensure_ascii=False)


if __name__ == '__main__':
    mcp.run(transport='stdio')

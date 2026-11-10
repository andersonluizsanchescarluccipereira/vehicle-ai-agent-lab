"""API REST. Local, sem autenticação: NÃO publicar na internet."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from src.agent_graph import ask

app = FastAPI(title='Vehicle AI Agent Lab', version='1.0.0',
              description='LangGraph + LangChain + Ollama + MCP + OKF + OpenSearch')


class SearchRequest(BaseModel):
    pergunta: str = Field(min_length=1, max_length=500)
    modo: Literal['bm25', 'vector', 'hybrid'] = 'hybrid'
    limite: int = Field(default=5, ge=1, le=5)
    sem_llm: bool = False


@app.get('/health')
def health() -> dict:
    return {'status': 'alive', 'message': 'Processo FastAPI operacional; dependências não verificadas'}


@app.post('/api/v1/veiculos/consultar')
async def consultar(request: SearchRequest) -> dict:
    try:
        return await ask(request.pergunta, request.modo, request.limite, request.sem_llm)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        # Exibe apenas um erro genérico; detalhes técnicos devem ir aos logs locais.
        import logging
        logging.getLogger(__name__).exception('Falha na consulta do agente')
        raise HTTPException(status_code=503, detail='Serviço local indisponível; consulte os logs.') from exc

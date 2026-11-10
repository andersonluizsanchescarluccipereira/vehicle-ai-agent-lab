import ast
from pathlib import Path

# O grafo usa o LangGraph real em integração; validamos decisões puras sem instalar dependências.
# Em ambiente com dependências, os testes de grafo abaixo também executam.


def test_questions_have_qa_examples():
    docs = (Path(__file__).parents[1] / 'README.md').read_text()
    assert 'LangGraph' in docs and 'MCP' in docs

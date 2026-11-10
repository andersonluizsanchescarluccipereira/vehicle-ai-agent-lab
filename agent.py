"""CLI principal: python agent.py --pergunta '...' --modo hybrid [--sem-llm]."""
import argparse
import asyncio
import json
from src.agent_graph import ask


def main():
    parser = argparse.ArgumentParser(description='Vehicle AI Agent — LangGraph, MCP, OKF, OpenSearch, Ollama')
    parser.add_argument('--pergunta', default='Compare três SUVs automáticos de até R$ 100 mil para minha família')
    parser.add_argument('--modo', choices=['bm25', 'vector', 'hybrid'], default='hybrid')
    parser.add_argument('--limite', type=int, choices=[1, 2, 3, 4, 5], default=5)
    parser.add_argument('--sem-llm', action='store_true')
    args = parser.parse_args()
    result = asyncio.run(ask(args.pergunta, args.modo, args.limite, args.sem_llm))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

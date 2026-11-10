#!/usr/bin/env bash
set -u
printf '\n== Ambiente ==\n'
python3 --version || true
podman --version || true
podman-compose --version || true
printf '\n== Memória / disco ==\n'
free -h || true
df -h / || true
printf '\n== Ollama ==\n'
curl --max-time 5 -sSf http://localhost:11434/api/tags | head -c 500 || true
printf '\n\n== OpenSearch ==\n'
curl --max-time 5 -sSf http://localhost:9200/_cluster/health?pretty || true
printf '\n\n== Containers ==\n'
podman ps || true

#!/usr/bin/env bash
# Único comando para levantar el sistema completo (Redis + API + worker).
#   ./run.sh
#
# La primera vez, si falta .env, lo crea desde .env.example y te pide completar las API keys antes de
# seguir. La ingesta del RAG (Pinecone) es automática al arrancar la API (ver app/rag/startup.py).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Se creó .env a partir de .env.example. Completá las API keys (GOOGLE_API_KEY, PINECONE_API_KEY,"
  echo "LANGSMITH_API_KEY son obligatorias; TAVILY_API_KEY es opcional) y volvé a correr ./run.sh"
  exit 1
fi

docker compose up --build

![python](https://img.shields.io/badge/Python-3.11.14-3776AB.svg?style=flat&logo=python)
![postgres](https://img.shields.io/badge/PostgreSQL-17.0-4169E1.svg?style=flat&logo=postgresql)
![docker](https://img.shields.io/badge/Docker-28.5.1-2496ED.svg?style=flat&logo=docker)
![elasticsearch](https://img.shields.io/badge/Elasticsearch-9.1.1-005571.svg?style=flat&logo=elasticsearch)
![kibana](https://img.shields.io/badge/Kibana-9.1.1-005571.svg?style=flat&logo=elasticsearch)
![pydantic](https://img.shields.io/badge/Pydantic-2.12.2-E92063.svg?style=flat&logo=pydantic)
![brave](https://img.shields.io/badge/Brave-FB542B.svg?style=flat&logo=brave)

## Introduction
The Huberman Lab podcast consistently ranks in the top 5 across Apple and Spotify in the Health, Fitness, and Science categories, with over 7 million YouTube subscribers. While unequivocally popular, the episodes are long and often not easy to digest. Each episode averages 120 minutes and the longest episode, featuring Dr. Andy Galpin on "Optimal Protocols to Build Strength and Muscle," runs 279 minutes — that’s over 4.5 hours!

The podcast offers both knowledge and valuable tools that could improve our lives, but this content is hidden in excessively long episodes. An agentic system addresses this gap by acting as a personalized coach that extracts content from the podcast's knowledge base to recommend actionable tools that users can immediately implement. By grounding recommendations in evidence-based research and expert interviews from the podcast, the system ensures that recommendations are both scientifically sound and immediately actionable.

## Setup
- Faster-Whisper supports Python 3.8–3.11 (no 3.12 yet), so use one of those versions; 3.11.x is recommended for best compatibility with this project.

- `docker-compose.yaml` starts Postgres + pgAdmin, Elasticsearch + Kibana.
  ```bash
  docker compose up -d
  ```
  Postgres listens on `localhost:5434`; pgAdmin on `http://localhost:8080` (admin/admin). Elasticsearch is available at `http://localhost:9200`, Kibana at `http://localhost:5601`.
- uv manages Python packages and environments (`pyproject.toml`/`uv.lock` are tracked in-repo). Install with `pip install uv` and initialize with `uv init`.
- API keys are managed via `.direnv` (for example, an `.envrc` file loaded by direnv); ensure your environment files are present before sending requests to an LLM.

## Ingestion
<img src='diagrams/ingestion.png'>

- Audio files are downloaded from RSS, transcribed with the Faster-Whisper tiny model, chunked via a sliding window, and embedded with Sentence Transformers. 
- RSS metadata and transcripts can be stored either locally or in PostgreSQL, while embeddings can be stored in Elasticsearch or Qdrant. 
- [Further ingestion instructions](ingestion/README.md).

## Agent


## Test and Evaluation

## Logging and Monitoring

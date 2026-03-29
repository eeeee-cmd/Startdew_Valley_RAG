# Stardew Valley RAG

This repository is a team project scaffold for a Retrieval-Augmented Generation (RAG) chatbot focused on the Stardew Valley Wiki.

The current repository is intentionally lightweight. It provides the project structure, module layout, data folders, and a raw wiki dataset snapshot so the team can build the rest of the pipeline in a consistent way.

## Project Purpose

The long-term goal of this project is to build a conversational RAG system that can answer Stardew Valley questions using grounded information from the public Stardew Valley Wiki.

Planned system stages:

- data extraction from the Stardew Valley Wiki
- preprocessing and normalization
- chunking for retrieval
- embeddings generation
- vector database storage
- retrieval pipeline
- agent orchestration
- frontend or user interface

## Current Repository Status

What is currently included:

- project folder structure for all planned modules
- placeholder files for future implementation
- raw extracted wiki data in `data/raw/`
- a root `README.md` describing the scaffold

What is not currently included:

- finalized extraction code
- preprocessing logic
- chunking logic
- embeddings code
- vector database integration
- retrieval implementation
- agent implementation
- frontend implementation

## Repository Structure

```text
Stardew_valley_RAG/
├── .env.example
├── .gitignore
├── README.md
├── config.py
├── main.py
├── requirements.txt
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
├── notebooks/
├── src/
│   ├── extraction/
│   ├── preprocessing/
│   ├── chunking/
│   ├── embeddings/
│   ├── vectorstore/
│   ├── retrieval/
│   ├── agent/
│   ├── frontend/
│   └── utils/
└── tests/
```

## Folder Guide

- `data/raw/`: source data snapshots collected from the wiki
- `data/interim/`: temporary or cleaned outputs created during preprocessing
- `data/processed/`: downstream outputs prepared for retrieval or modeling
- `src/extraction/`: future wiki extraction logic
- `src/preprocessing/`: future text cleaning and normalization logic
- `src/chunking/`: future chunk creation logic
- `src/embeddings/`: future embedding generation logic
- `src/vectorstore/`: future vector database integration
- `src/retrieval/`: future retrieval pipeline logic
- `src/agent/`: future RAG orchestration and conversation logic
- `src/frontend/`: future user interface code
- `src/utils/`: shared helpers used across modules
- `tests/`: unit tests and integration tests
- `docs/`: project documentation
- `notebooks/`: experiments and exploration work

## Data Notes

The repository currently contains raw Stardew Valley Wiki extraction data under `data/raw/`.

Recommended data workflow:

1. keep `data/raw/` as the original source snapshot
2. write cleaned or transformed outputs into `data/interim/`
3. write retrieval-ready artifacts into `data/processed/`

## Suggested Team Workflow

- keep each module focused on one responsibility
- avoid mixing extraction, preprocessing, retrieval, and agent logic in the same file
- use `src/` as the main implementation area
- use `tests/` for validation as code is added
- document major design decisions in `docs/`

## Setup

Basic environment setup:

```bash
pip install -r requirements.txt
```

Optional environment file setup:

```bash
copy .env.example .env
```

## Next Steps

Recommended implementation order:

1. finalize extraction module
2. add preprocessing and cleaning pipeline
3. implement chunking strategy
4. add embeddings and vector storage
5. implement retrieval and answer generation
6. connect the system to a simple frontend

## Notes For Teammates

This repository is meant to be extended gradually. The current state should be treated as a clean starting point rather than a finished pipeline.

If the team adds working code later, update this README so it stays aligned with the actual state of the repository.

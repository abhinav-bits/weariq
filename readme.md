WearIQ
AI-powered backend system for clothing inventory understanding, semantic search, and generation workflows

🧠 What is WearIQ?
WearIQ is a developer-first platform for building AI-driven clothing applications.

It provides the core infrastructure to:

ingest clothing inventory from images

extract structured attributes

enable semantic product search

run async AI workflows (embeddings, generation, etc.)

This repo focuses on backend systems, data pipelines, and AI orchestration, not UI.

🎯 Scope (MVP)
Current implementation focuses on:

inventory ingestion (images + optional metadata)

attribute extraction pipeline

embedding generation

vector indexing

semantic + filtered search

async job processing

Out of scope (for now):

full virtual try-on

video generation

production-grade marketplace features

🧱 Architecture
Modular monolith + async workers.

apps/
  api/        → REST API (inventory, search, jobs)
  workers/    → async processing (AI pipelines)

packages/
  shared/     → types, utilities

infra/
  db          → PostgreSQL
  vector      → Redis / Qdrant (TBD)
  storage     → object storage (S3 compatible)
  queue       → Redis-based queue
Core components
API Layer

inventory CRUD

search endpoints

job orchestration

Worker Layer

attribute extraction

embedding generation

indexing

generation tasks (future)

Data Layer

PostgreSQL → metadata

Vector DB → embeddings

Object storage → images

🔄 Data Flow
Inventory ingestion
upload → store image → create item → enqueue job
        ↓
   extract attributes → generate embedding → index
Search
query → embedding → vector search → filter → rank → results
📦 Project Structure
weariq/
├── apps/
│   ├── api/
│   └── workers/
├── packages/
│   └── shared/
├── infra/
├── docs/
└── README.md
⚙️ Setup (WIP)
Requirements
Node.js / Python (depending on service)

PostgreSQL

Redis

Object storage (S3 or compatible)

Local setup (placeholder)
git clone https://github.com/<your-org>/weariq
cd weariq
Detailed setup will be added per service.

🔌 Key Modules (Planned)
inventory

search

embedding

extraction

jobs

media

Each module should:

be isolated

expose clear interfaces

support async processing

🧵 Async Jobs
All heavy operations are async:

attribute extraction

embedding generation

indexing

future: image generation

Job lifecycle
pending

running

success

failed

🧪 Development Guidelines
prefer simple, explicit APIs

avoid early microservices

keep modules decoupled

design for reprocessing (idempotent jobs)

log everything important

🚦 Status
Early stage — architecture and core pipelines under active development.

📌 Notes
Search quality is a core focus area

Metadata extraction accuracy directly impacts results

System is designed to evolve toward try-on and generation workflows
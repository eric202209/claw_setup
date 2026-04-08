# OpenClaw + Ollama hybrid architecture

The Ultimate OpenClaw + Ollama Hybrid Architecture Deployment Guide

This document documents how we overcame Docker network isolation, permission persistence, 
and integrated the advanced QMD search engine from scratch.

---

## Build llama.cpp (on HOST)

```bash
sudo apt-get install -y git cmake build-essential patchelf

git clone https://github.com/ggerganov/llama.cpp \
  /home/your_folder_path/llama.cpp
cd /home/your_folder_path/llama.cpp

cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120

cmake --build build --config Release -j $(nproc)

/home/your_folder_path/llama.cpp/build/bin/llama-server --version

```

---

## Download the Model (on HOST)

```bash
pip install huggingface_hub
mkdir -p /home/your_folder_path/models

huggingface-cli download \
  unsloth/Qwen3.5-35B-A3B-GGUF \
  --include "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf" \
  --local-dir /home/your_folder_path/models/

ls -lh /home/your_folder_path/models/
# Should show ~20 GB file
```

**Why this model?**
- `UD` = Unsloth Dynamic quantisation
- ~20 GB vs ~37 GB for full-size quant
- MoE architecture: only 3B params activated per token → fast

---

## Create llama-proxy.py (On Host)

Sample file as ai-gateway/proxy/llama-proxy.py
(Build your own version in the future)

---

## Prepare Host Directories

```bash
mkdir -p /home/your_file_path/workspace
mkdir -p /home/your_file_path/memory
```

If rebuilding from existing container, copy data out first:

```bash
docker cp your_folder_name:/root/.openclaw/workspace /home/your_file_path/
docker cp your_folder_name:/root/.openclaw/memory    /home/your_file_path/
```

---

## Get Gemini API key 

free at: https://aistudio.google.com/apikey
> Free tier: 100 RPD, 5 RPM for Gemini 2.5 Flash

```bash
| `:ro` on llama.cpp and qwen_claude | Read-only, agent cannot modify model files |
| `-e GEMINI_API_KEY` | Enables Gemini routing in proxy |
| workspace + memory mounts | Survive container rebuild |
```

---

### Make sure the file structure the same as this REPO

---

## 🏗️ System Architecture Overview

This system employs a hybrid deployment strategy, balancing isolation and performance:

• Container (Docker): Executes OpenClaw (ai-agent) to handle dialogue logic and UI.

• Host: Executes Ollama to provide embedding computing power and runs QMD for ultra-fast file retrieval.

---

## 🚀 Common Use Command Prompt

```bash

docker compose build

docker compose down
docker compose up -d
docker compose logs -f ai-agent

docker start ai-agent
docker restart ai-agent

docker exec -it ai-agent /bin/bash
```

---

### 📦 Phase One: Docker Compose Environment Configuration

To ensure data is not lost upon container restarts, we configured critical persistent volumes and network bridging.

Key snippets of docker-compose.yml

services:
ai-agent:
image: your_name:latest
container_name: ai-agent

environment:
- OPENCLAW_GATEWAY_TOKEN= your_secure_token

extra_hosts:
- "host.docker.internal:host-gateway" # Key: Allows containers to access the host machine

volumes:
- ./your/file/path/workspace:/root/.openclaw/workspace
- ./your/file/path/memory:/root/.openclaw/memory
- ./your/file/path/config/openclaw.json:/root/.openclaw/openclaw.json
- ./your/file/path/redis:/var/lib/redis # Redis persistence

ports: 
- "8000:8000"

---

### ⚙️ Phase Two: Host Machine Ollama Configuration

To allow the Agent within Docker to use the host machine's GPU for embedding, the Ollama listening settings must be adjusted.

1. Modify service settings: Execute `sudo systemctl edit ollama.service` and add:

Ini, TOML

[Service]

Environment="OLLAMA_HOST=0.0.0.0"

2. Restart the service:

```bash
sudo systemctl daemon-reload

sudo systemctl restart ollama
```

---

### 🔍 Phase Three: QMD v2 Advanced Search (Host Side)

Install QMD on the host machine to leverage the powerful computing capabilities for semantic search and reranking.

1. Installation and Configuration:

```bash
npm install -g @tobilu/qmd
qmd collection add /path/to/your/vault --name vault
```

2. Indexing (using GPU acceleration):

```bash
QMD_EMBEDDING_MODEL=nomic-embed-text qmd embed
```
3. Fast Terminal Search:

```bash
◦ qmd query "question": Deep search using Reranker.

◦ qmd vsearch "sentence": Pure vector similarity search.
```

---

After ai-agent / ai-gateway container was built,

docker exec -it ai-agent /bin/bash

## Install Tools Inside Container

```bash
apt-get update && apt-get install -y \
  curl wget git python3 python3-pip \
  build-essential openssh-client

# Node.js 22
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs

# pnpm
npm install -g pnpm

# GitHub CLI
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] \
  https://cli.github.com/packages stable main" \
  | tee /etc/apt/sources.list.d/github-cli.list
apt-get update && apt-get install -y gh

# Verify
git --version && gh --version && pnpm --version && node --version
```

---

## Setup GitHub Agent Account

```bash
git config --global user.name "your_user_name"
git config --global user.email "your_email"

gh auth login --with-token <<< "YOUR_GITHUB_TOKEN"
gh auth setup-git
gh auth status
```

Fix git safe directory (required when workspace is a mounted volume):

```bash
git config --global --add safe.directory /root/.openclaw/workspace
```

---

## Generate openclaw Token

```bash
openssl rand -hex 32
# Save this output — used everywhere as YOUR_TOKEN
```

You can follow OpenClaw official install guide to get your Gateway Token 

---

## Create openclaw Config (openclaw.json)
sample file as openclaw.json
put that file in ~/.openclaw

---

## Every Session Startup

```bash
docker compose down    - Do it first when you update docker-compose.yml, DockerFile, llama-proxy.py...etc
docker compose up -d
docker start ai-agent  - Launch the system
docker compose logs -f ai-agent  - Check it's working with no error
docker exec -it ai-agent /bin/bash   - Enter the root
```

---

## Keyword Reference

| Keyword | Where | Effect |
|---------|-------|--------|
| `[think]` | Start of message | Enables Qwen3.5 thinking mode |
| `[gemini]` | Anywhere in message | Routes to Gemini 2.5 Flash |
| `[autoresearch]` | Search for that then you know how to do |

**example:**
```
[gemini] review the PropertyCard component for bugs
```

---

## 🛠️ Troubleshooting

During the setup process, we successfully resolved the following core issues:

1. Container cannot connect to Ollama (Connection Refused)

• Symptom: Executing curl host.docker.internal:11434 has no effect.

• Solution:

◦ Ensure OLLAMA_HOST is set to 0.0.0.0.


2. QMD cannot find the Embedding model.

• Symptom: Executing `qmd embed` results in an error.

• Solution:

◦ Ensure that the host machine Ollama has downloaded `nomic-embed-text`.

◦ Specify the environment variable `QMD_EMBEDDING_MODEL=nomic-embed-text` during execution.


3. Recommended final state:

- Suggest you use openclaw@2026.3.11 or @2026.3.13, more stable version

- Node 22 LTS

- npm install -g openclaw@2026.3.13 (if that’s now verified good in your environment)

- Compose command uses openclaw gateway

---

## 💡 Summary and Follow-up

Follow-up recommendations:

• Regularly run `qmd update` to keep the index up-to-date.

• Utilize OpenClaw's workspace mount points to allow agents to directly read and modify your project code.

---

## Future: NemoClaw Migration

When NemoClaw becomes stable:

```bash
# 1. Backup workspace
docker cp your_folder_name:/root/.openclaw/workspace \
  /home/your_username/Documents/backup/pre-nemoclaw-$(date +%Y%m%d)/

# 2. Install NemoClaw (when available)
curl -fsSL https://nvidia.com/nemoclaw.sh | bash
nemoclaw onboard
```

All workspace files (SOUL.md, AGENTS.md, TODO.md, projects/, memory/) are compatible.

---

## License

MIT
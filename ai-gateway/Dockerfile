FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

# Install Python dependencies
RUN apt-get update && apt-get install -y     python3     python3-pip     curl     && rm -rf /var/lib/apt/lists/*

# Pre-installed yaml package required for proxy (solves previous PEP 668 problem)
RUN pip install pyyaml --break-system-packages

WORKDIR /opt/ai-gateway

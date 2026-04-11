FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    curl redis-server build-essential git python3 jq \
    pkg-config libpixman-1-dev libcairo2-dev libpango1.0-dev \
    libjpeg-dev libgif-dev librsvg2-dev && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g pnpm openclaw@2026.4.10

WORKDIR /root

EXPOSE 18789

CMD service redis-server start && \
    export HOST=0.0.0.0 && \
    openclaw gateway --port 18789 --allow-unconfigured


# ─── builder stage ───────────────────────────────────────────────────────────
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS builder

# Set timezone non-interactively
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt update -y \
 && apt install -y curl git \
 && curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh \
 && rm -rf /var/lib/apt/lists/*

ENV PATH="${PATH}:/root/.local/bin"

# Add the deadsnakes PPA for Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    gnupg \
    lsb-release \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
 && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN touch README.md

# Copy source code so uv can install the local package
COPY triel ./triel

# uv creates a venv and installs deps (including the local package)
RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
RUN --mount=type=ssh uv sync --no-group dev -v

COPY install_lms.sh ./
COPY README.md logging.conf ./

# Install language models
RUN chmod +x install_lms.sh && ./install_lms.sh

# ─── runtime stage ────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS runtime

# Set timezone non-interactively
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt update -y \
 && apt install -y curl git \
 && curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh \
 && rm -rf /var/lib/apt/lists/*

ENV PATH="${PATH}:/root/.local/bin"

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    gnupg \
    lsb-release \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
 && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the entire app including venv from builder
COPY --from=builder /app /app

# Configuration is provided via environment variables (see .env.example)
# Use docker run -e or docker-compose to set them
CMD ["uv", "run", "triel"]

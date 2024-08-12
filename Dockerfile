FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS builder

RUN apt update -y && apt upgrade -y && apt install curl git -y

RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    git \
    gnupg \
    lsb-release

# Add the deadsnakes PPA for Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.10 and necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip

# Update alternatives to set Python 3.10 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN curl -sSL https://install.python-poetry.org | python - --version 1.8.3

ENV PATH="${PATH}:/root/.local/bin"

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_EXPERIMENTAL_SYSTEM_GIT_CLIENT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN touch README.md

COPY lm_service ./lm_service

RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
RUN --mount=type=ssh poetry install --no-interaction -vvv --without dev
RUN poetry run python -m spacy download en_core_web_lg
RUN poetry run python -m spacy download en_core_web_trf
RUN poetry run python -m coreferee install en
COPY run ./run
COPY README.md logging.conf ./

CMD ["poetry", "run", "python", "run/serve.py", "--wsgi-self", "self.json", "--entity-linker-config", "el_config.yaml"]

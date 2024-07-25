FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS builder

RUN apt update -y && apt upgrade -y && apt install curl git -y
RUN apt install -y python3.10
RUN curl -sSL https://install.python-poetry.org | python - --version 1.8.3

#ENV PATH="${PATH}:/root/.local/bin"
#
#ENV POETRY_NO_INTERACTION=1 \
#    POETRY_VIRTUALENVS_IN_PROJECT=1 \
#    POETRY_VIRTUALENVS_CREATE=1 \
#    POETRY_EXPERIMENTAL_SYSTEM_GIT_CLIENT=1 \
#    POETRY_CACHE_DIR=/tmp/poetry_cache
#
#WORKDIR /app
#
#COPY pyproject.toml poetry.lock ./
#RUN touch README.md
#
#COPY task_manager ./task_manager
#
#RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
#RUN --mount=type=ssh poetry install --no-interaction -vvv --without dev
#COPY conf ./conf
#COPY run ./run
#COPY README.md logging.conf logging.debug.conf ./
#
#CMD ["poetry", "run", "python", "run/gg_server.py", "--wsgi-self", "conf/server/self.json", "--schema-path", "conf/schema/kg_v3b.yaml", "--db-conf-path", "db_conf.json", "--user-db-conf-path", "db_conf.user.json", "--wsgi-re-config", "re.json", "--db", "kg_v3"]

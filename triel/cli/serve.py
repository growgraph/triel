import io
import json
import logging
import logging.config
import pathlib
import traceback
from typing import Any

import click
import spacy
from flask import Flask, jsonify, request
from flask_compress import Compress
from flask_cors import cross_origin
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_restful import Api
from pydantic import BaseSettings, Field
from suthing import FileHandle
from waitress import serve

from triel.linking.onto import (
    EntityLinkerFailed,
    EntityLinkerManager,
    EntityLinkerTypeNotAvailable,
)
from triel.top import (
    cast_response_entity_representation,
    cast_response_redux,
    text_to_graph_mentions_entities,
)

app = Flask(__name__)
Compress(app)
api = Api(app)

logger = logging.getLogger(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    # default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)


@app.route("/ping")
@limiter.exempt
def ping():
    return "PONG"


def get_exception_traceback_str(exc: Exception) -> str:
    # Ref: https://stackoverflow.com/a/76584117/
    file = io.StringIO()
    traceback.print_exception(exc, file=file)
    return file.getvalue().rstrip()


# ─── Configuration Models ──────────────────────────────────────────────────────


class ServerConfig(BaseSettings):
    """Server configuration (TRIEL_SERVER_*)."""

    class Config:
        env_prefix = "TRIEL_SERVER_"
        case_sensitive = False

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8592, description="Server port")
    threads: int = Field(default=8, description="Number of concurrent threads")
    debug: bool = Field(default=False, description="Enable debug logging")


class WSGIPathsConfig(BaseSettings):
    """WSGI API paths configuration (TRIEL_WSGI_PATHS_*)."""

    class Config:
        env_prefix = "TRIEL_WSGI_PATHS_"
        case_sensitive = False

    parse: str = Field(default="/v3/parse", description="Parse endpoint path")
    parse_entities: str = Field(
        default="/v3/parse/entities", description="Parse entities endpoint path"
    )
    parse_detailed: str = Field(
        default="/v3/parse/detailed", description="Parse detailed endpoint path"
    )

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format expected by existing code."""
        return {
            "parse": self.parse,
            "parse/entities": self.parse_entities,
            "parse/detailed": self.parse_detailed,
        }


class WSGIConfig(BaseSettings):
    """WSGI configuration (TRIEL_WSGI_*)."""

    class Config:
        env_prefix = "TRIEL_WSGI_"
        case_sensitive = False

    # Option 1: JSON string in env var
    config_json: str | None = Field(
        default=None,
        description="WSGI config as JSON string (alternative to config_file)",
    )

    # Option 2: File path (backward compatible)
    config_file: pathlib.Path | None = Field(
        default=None,
        description="Path to WSGI config file",
    )

    protocol: str = Field(default="http", description="Protocol")
    db_type: str = Field(default="wsgi", description="Database type")
    ip_addr: str = Field(default="localhost", description="IP address")
    paths: WSGIPathsConfig = Field(
        default_factory=WSGIPathsConfig, description="API paths"
    )

    @classmethod
    def from_file(cls, file_path: pathlib.Path | None = None) -> "WSGIConfig":
        """Load from JSON string, file, or environment variables."""
        # First, load from env vars to check for config_json
        config = cls()

        # Priority: JSON string > file parameter > config_file env var > env vars
        if config.config_json:
            config_dict = json.loads(config.config_json)
            if "paths" in config_dict:
                config_dict["paths"] = WSGIPathsConfig(**config_dict["paths"])
            return cls(**config_dict)
        elif file_path and file_path.exists():
            config_dict = FileHandle.load(fpath=file_path)
            if "paths" in config_dict:
                config_dict["paths"] = WSGIPathsConfig(**config_dict["paths"])
            return cls(**config_dict)
        elif config.config_file and config.config_file.exists():
            config_dict = FileHandle.load(fpath=config.config_file)
            if "paths" in config_dict:
                config_dict["paths"] = WSGIPathsConfig(**config_dict["paths"])
            return cls(**config_dict)

        return config


class ModelConfig(BaseSettings):
    """Model configuration (TRIEL_MODEL_*)."""

    class Config:
        env_prefix = "TRIEL_MODEL_"
        case_sensitive = False

    gpu: bool = Field(default=True, description="Load spaCy models to GPU")
    spacy_model: str = Field(default="en_core_web_trf", description="spaCy model name")
    rules_file: str = Field(
        default="prune_noun_compound_v3.yaml",
        description="Rules configuration file name",
    )
    rules_path: str = Field(
        default="triel.config", description="Rules configuration path"
    )


class LinkerConfig(BaseSettings):
    """Entity linker configuration (TRIEL_LINKER_*)."""

    class Config:
        env_prefix = "TRIEL_LINKER_"
        case_sensitive = False

    # Option 1: JSON string in env var
    config_json: str | None = Field(
        default=None,
        description="Linker config as JSON string (alternative to config_file)",
    )

    # Option 2: File path (backward compatible)
    config_file: pathlib.Path | None = Field(
        default=None, description="Path to entity linker config file"
    )
    host_override: str | None = Field(
        default=None, description="Override host for all linkers"
    )

    def load_config(self, file_override: pathlib.Path | None = None) -> dict[str, Any]:
        """Load entity linker configuration from JSON string or file.

        Priority: file_override > JSON string > file path
        """
        # Priority: CLI file override > JSON string > file path
        if file_override and file_override.exists():
            el_config = FileHandle.load(fpath=file_override)
        elif self.config_json:
            el_config = json.loads(self.config_json)
        elif self.config_file and self.config_file.exists():
            el_config = FileHandle.load(fpath=self.config_file)
        else:
            raise ValueError(
                "Either --entity-linker-config, TRIEL_LINKER_CONFIG_JSON, or TRIEL_LINKER_CONFIG_FILE must be set"
            )

        # Apply host override if provided
        if self.host_override:
            for linker in el_config.get("linkers", []):
                linker["host"] = self.host_override

        return el_config


class AppConfig(BaseSettings):
    """Main application configuration."""

    class Config:
        env_prefix = "TRIEL_"
        case_sensitive = False

    server: ServerConfig = Field(default_factory=ServerConfig)
    wsgi: WSGIConfig = Field(default_factory=WSGIConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    linker: LinkerConfig = Field(default_factory=LinkerConfig)

    @classmethod
    def from_files(
        cls,
        wsgi_config_file: pathlib.Path | None = None,
        linker_config_file: pathlib.Path | None = None,
    ) -> "AppConfig":
        """Create configuration from files and environment variables."""
        config = cls()

        # Load WSGI config from file if provided
        if wsgi_config_file:
            config.wsgi = WSGIConfig.from_file(wsgi_config_file)

        # Set linker config file if provided
        if linker_config_file:
            config.linker.config_file = linker_config_file

        return config


# ─── Main Application ──────────────────────────────────────────────────────────


@click.command()
@click.option(
    "--wsgi-config",
    type=click.Path(path_type=pathlib.Path, exists=True),
    help="Path to WSGI config file (JSON/YAML). Overrides TRIEL_WSGI_CONFIG_JSON and TRIEL_WSGI_CONFIG_FILE.",
)
@click.option(
    "--entity-linker-config",
    type=click.Path(path_type=pathlib.Path, exists=True),
    help="Path to entity linker config file (JSON/YAML). Overrides TRIEL_LINKER_CONFIG_JSON and TRIEL_LINKER_CONFIG_FILE.",
)
@click.option(
    "--host",
    type=str,
    help="Override server host (overrides TRIEL_SERVER_HOST)",
)
@click.option(
    "--port",
    type=int,
    help="Override server port (overrides TRIEL_SERVER_PORT)",
)
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--threads", type=int, help="Number of concurrent threads")
@click.option("--gpu/--no-gpu", default=None, help="Enable/disable GPU")
def main(
    wsgi_config: pathlib.Path | None,
    entity_linker_config: pathlib.Path | None,
    host: str | None,
    port: int | None,
    debug: bool,
    threads: int | None,
    gpu: bool | None,
):
    """Triel server - Convert text to triples with entity linking."""
    # Load configuration from files and environment variables
    app_config = AppConfig.from_files(
        wsgi_config_file=wsgi_config,
        linker_config_file=entity_linker_config,
    )

    # Apply command-line overrides
    if host is not None:
        app_config.server.host = host
    if port is not None:
        app_config.server.port = port
    if debug:
        app_config.server.debug = debug
    if threads is not None:
        app_config.server.threads = threads
    if gpu is not None:
        app_config.model.gpu = gpu

    # Configure logging
    debug_option = ".debug" if app_config.server.debug else ""
    logger_conf = f"logging{debug_option}.conf"
    logging.config.fileConfig(logger_conf, disable_existing_loggers=False)
    if app_config.server.debug:
        logger.debug("Debug mode enabled")

    # Load entity linker configuration
    el_config = app_config.linker.load_config(file_override=entity_linker_config)
    elm = EntityLinkerManager.from_dict(el_config)

    # Load rules
    rules = FileHandle.load(app_config.model.rules_path, app_config.model.rules_file)

    # Configure GPU if enabled
    if app_config.model.gpu:
        spacy.prefer_gpu()

    # Load spaCy model
    nlp = spacy.load(app_config.model.spacy_model)
    nlp.add_pipe("coreferee")

    def work(request0):
        json_data = request0.json
        text = json_data["text"]
        response = text_to_graph_mentions_entities(text, nlp, rules, elm)
        return response

    # Get paths from config
    paths = app_config.wsgi.paths.to_dict()

    @app.route(paths["parse"], methods=["POST"])
    @app.route(paths["parse/detailed"], methods=["POST"])
    @limiter.limit("10/second", override_defaults=False)
    @cross_origin()
    def parse():
        if request.method == "POST":
            try:
                response = work(request)
            except EntityLinkerFailed as e:
                logger.error(f"EntityLinkerFailed : {e}")
                return {"error": get_exception_traceback_str(e)}, 502
            except EntityLinkerTypeNotAvailable as e:
                logger.error(f"EntityLinkerTypeNotAvailable : {e}")
                return {"error": get_exception_traceback_str(e)}, 501
            except Exception as e:
                logger.error(f"Exception: {e}")
                return {"error": get_exception_traceback_str(e)}, 500

            response_cast = cast_response_redux(response)
            response_dictlike = response_cast.to_dict()
            jy = jsonify(response_dictlike)
            return jy, 200

    @app.route(paths["parse/entities"], methods=["POST"])
    @limiter.limit("10/second", override_defaults=False)
    @cross_origin()
    def parse_entities():
        if request.method == "POST":
            try:
                response = work(request)
            except EntityLinkerFailed as e:
                return {"error": get_exception_traceback_str(e)}, 502
            except EntityLinkerTypeNotAvailable as e:
                return {"error": get_exception_traceback_str(e)}, 501
            except Exception as e:
                return {"error": get_exception_traceback_str(e)}, 500

            response_cast = cast_response_entity_representation(response)
            response_dictlike = response_cast.to_dict()
            jy = jsonify(response_dictlike)
            return jy, 200

    logger.info(f"Starting server on {app_config.server.host}:{app_config.server.port}")
    logger.info("REEL model loaded")
    serve(
        app,
        host=app_config.server.host,
        port=app_config.server.port,
        threads=app_config.server.threads,
    )


if __name__ == "__main__":
    main()

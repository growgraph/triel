import io
import logging
import logging.config
import pathlib
import traceback

import click
import spacy
from flask import Flask, jsonify, request
from flask_compress import Compress
from flask_cors import cross_origin
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_restful import Api
from suthing import ConfigFactory, FileHandle
from waitress import serve

from lm_service.linking.onto import (
    EntityLinkerFailed,
    EntityLinkerManager,
    EntityLinkerTypeNotAvailable,
)
from lm_service.top import (
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


@click.command()
@click.option(
    "--wsgi-self",
    type=click.Path(path_type=pathlib.Path),
    help="config to connect to db",
)
@click.option("--debug", is_flag=True, default=False, help="logging at debug level")
@click.option(
    "--entity-linker-config",
    help="entity linker config as json or yaml",
    type=click.Path(path_type=pathlib.Path),
)
@click.option("--host", type=click.STRING, default=None)
@click.option("--threads", type=int, default=8, help="number of concur threads")
@click.option("--gpu", help="load spacy models to gpu", is_flag=True, default=True)
@click.option("--debug", is_flag=True, default=False, help="logging at debug level")
def main(wsgi_self, entity_linker_config, host, debug, threads, gpu):
    debug_option = ".debug" if debug else ""
    logger_conf = f"logging{debug_option}.conf"
    logging.config.fileConfig(logger_conf, disable_existing_loggers=False)
    logger.debug("debug is on")

    wsgi_config = FileHandle.load(fpath=wsgi_self)
    wsgi_re = ConfigFactory.create_config(dict_like=wsgi_config)
    rules = FileHandle.load("lm_service.config", "prune_noun_compound_v3.yaml")

    el_config = FileHandle.load(fpath=entity_linker_config)
    if host is not None:
        for c in el_config["linkers"]:
            c["host"] = host
    elm = EntityLinkerManager.from_dict(el_config)

    if gpu:
        spacy.prefer_gpu()

    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    def work(request0):
        logger.info(request0)
        logger.info(request0.json)
        json_data = request0.json
        text = json_data["text"]
        response = text_to_graph_mentions_entities(text, nlp, rules, elm)
        return response

    @app.route(wsgi_re.paths["parse"], methods=["POST"])
    @app.route(wsgi_re.paths["parse/detailed"], methods=["POST"])
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

            response_jsonlike = cast_response_redux(response)
            jy = jsonify(response_jsonlike)
            return jy, 200

    @app.route(wsgi_re.paths["parse/entities"], methods=["POST"])
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
            response_jsonlike = cast_response_entity_representation(response)

            jy = jsonify(response_jsonlike)
            return jy, 200

    logger.info(f"wsgi: host {wsgi_re.host}")
    logger.info("REEL model loaded")
    serve(app, host=wsgi_re.host, port=wsgi_re.port, threads=threads)


if __name__ == "__main__":
    main()

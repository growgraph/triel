import argparse
import logging
import pkgutil

import coreferee
import spacy
import yaml
from flask import Flask, jsonify, request
from flask_restful import Api
from graph_cast.db.factory import ConfigFactory
from graph_cast.util import ResourceHandler
from waitress import serve

from lm_service.linking import EntityLinkerManager
from lm_service.top import cast_response_to_unfolded, text_to_rel_graph

app = Flask(__name__)
api = Api(app)

logger = logging.getLogger(__name__)


@app.route("/")
def hello_world():
    return "parse phrases into relations"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsgi-self", type=str)
    parser.add_argument(
        "-v",
        "--verbose",
        help="increase output verbosity",
        action="store_true",
    )

    parser.add_argument(
        "--entity-linker-config",
        type=str,
        help="entity linker config as json or yaml",
    )

    parser.add_argument(
        "--threads", type=int, default=8, help="number of concur threads"
    )

    args = parser.parse_args()

    logging.basicConfig(
        filename="lm_serve_re.log",
        format=(
            "%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s:"
            " %(message)s"
        ),
        level=logging.INFO if args.verbose else logging.ERROR,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )

    wsgi_config = ResourceHandler.load(fpath=args.wsgi_self)
    wsgi_re = ConfigFactory.create_config(args=wsgi_config)

    spacy.prefer_gpu()

    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")
    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound_v2.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    el_config = ResourceHandler.load(fpath=args.entity_linker_config)
    elm = EntityLinkerManager(el_config)

    @app.route(wsgi_re.path, methods=["POST"])
    def re():
        if request.method == "POST":
            logger.info(request)
            logger.info(request.json)
            json_data = request.json
            text = json_data["text"]
            response = text_to_rel_graph(text, nlp, rules, elm)
            response_jsonlike = cast_response_to_unfolded(
                response, cast_triple_version="v1"
            )

            jy = jsonify(response_jsonlike)
            return jy, 200

    logger.info(f" wsgi: host {wsgi_re.host}")
    logger.info(" re model loaded")
    serve(app, host=wsgi_re.host, port=wsgi_re.port, threads=args.threads)

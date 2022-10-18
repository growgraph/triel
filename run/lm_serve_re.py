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

from lm_service.top import text_to_rel_graph

app = Flask(__name__)
api = Api(app)

logger = logging.getLogger(__name__)


@app.route("/")
def hello_world():
    return "parse phrases into relations"


if __name__ == "__main__":
    logging.basicConfig(
        filename="lm_serve.log",
        format=(
            "%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s:"
            " %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filemode="w",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--wsgi-self", type=str)

    args = parser.parse_args()

    wsgi_config = ResourceHandler.load(fpath=args.wsgi_self)
    wsgi_re = ConfigFactory.create_config(args=wsgi_config)

    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")
    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound_v2.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)
    print(" re model loaded")

    @app.route(wsgi_re.path, methods=["POST"])
    def re():
        if request.method == "POST":
            logger.info(request)
            logger.info(request.json)
            json_data = request.json
            text = json_data["text"]
            response = text_to_rel_graph(text, nlp, rules)

            return jsonify(response), 200

    print(f" wsgi: host {wsgi_re.host}")
    app.run(port=wsgi_re.port, host=wsgi_re.host)

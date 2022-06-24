import argparse
import logging
import pkgutil
import yaml
import spacy
import coreferee
from flask import Flask, request, jsonify
from flask_restful import Api, reqparse

from graph_cast.util import ResourceHandler
from graph_cast.db.factory import ConfigFactory

from lm_service.relation import parse_relations_advanced, add_hash
from lm_service.preprocessing import normalize_input_text
from lm_service.graph import transform_advcl

app = Flask(__name__)
api = Api(app)

logger = logging.getLogger(__name__)


@app.route("/")
def hello_world():
    return "parse phrases into relations"


@app.route("/lm/re", methods=["POST"])
def re():
    if request.method == "POST":
        logger.info(request)
        logger.info(request.json)
        json_data = request.json
        text = json_data["phrase"]

        phrases = normalize_input_text(text, terminal_full_stop=False)
        phrases = [transform_advcl(nlp, p) for p in phrases]
        fragment = ". ".join(phrases)
        (
            graph,
            coref_graph,
            metagraph,
            triples_expanded,
            triples_proj,
        ) = parse_relations_advanced(fragment, nlp, rules)

        return jsonify({"triples": triples_proj}), 200


if __name__ == "__main__":
    logging.basicConfig(
        filename="lm_service.log",
        format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filemode="w",
        # stream=sys.stdout,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--wsgi-self", type=str)

    args = parser.parse_args()

    wsgi_config = ResourceHandler.load(fpath=args.wsgi_self)
    wsgi_re = ConfigFactory.create_config(args=wsgi_config)

    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")
    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)
    print(" re model loaded")

    @app.route(wsgi_re.path, methods=["POST"])
    def re_v2():
        if request.method == "POST":
            logger.info(request)
            logger.info(request.json)
            json_data = request.json
            text = json_data["phrase"]

            phrases = normalize_input_text(text, terminal_full_stop=False)
            phrases = [transform_advcl(nlp, p) for p in phrases]
            fragment = ". ".join(phrases)
            (
                graph,
                coref_graph,
                metagraph,
                triples_expanded,
                triples_proj,
            ) = parse_relations_advanced(fragment, nlp, rules)
            r = add_hash(triples_expanded, graph)
            return jsonify({"triples": r}), 200
    print(f" wsgi: host {wsgi_re.host}")
    app.run(port=wsgi_re.port, host=wsgi_re.host)

import spacy
import coreferee
import pkgutil
import yaml
from flask import Flask, request, jsonify
from flask_restful import Api, reqparse
from lm_service.relation import parse_relations_advanced
from lm_service.preprocessing import normalize_input_text
from lm_service.graph import transform_advcl
import logging

app = Flask(__name__)
api = Api(app)

logger = logging.getLogger(__name__)


@app.route("/")
def hello_world():
    return "score terms using lm"


@app.route("/lm/re", methods=["POST"])
def score():
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

        return jsonify({"result": triples_proj}), 200


if __name__ == "__main__":
    logging.basicConfig(
        filename="lm_service.log",
        format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filemode="w",
        # stream=sys.stdout,
    )

    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")
    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)
    print("models loaded")
    app.run(port=5001)

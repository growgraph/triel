from flask import Flask, request, jsonify
from flask_restful import Api, reqparse
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM
import logging

app = Flask(__name__)
api = Api(app)

logger = logging.getLogger(__name__)


@app.route("/")
def hello_world():
    return "score terms using lm"


@app.route("/lmscore", methods=["POST"])
def score():
    if request.method == "POST":
        logger.info(request)
        logger.info(request.json)
        json_data = request.json
        # phrases = ["Cut three zuccini then fry them",
        #            "John was cooking meat with vegetables",
        #            "Alexander drove his Honda for 100 miles each day"]
        # foci = ["zuccini", "John", "Honda"]
        # categories = ["person", "vegetable", "car"]
        phrases = json_data["phrases"]
        foci = json_data["foci"]
        categories = json_data["categories"]

        report = []
        for p, f in zip(phrases, foci):
            candl = [f"{f} is a {c}" for c in categories]
            r = classifier(p, candl)
            report += [r]

        return jsonify({"result": report}), 200


if __name__ == "__main__":
    logging.basicConfig(
        filename="lm_service.log",
        format=(
            "%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s:"
            " %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filemode="w",
        # stream=sys.stdout,
    )

    model = "facebook/bart-large-mnli"

    # model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base", output_hidden_states=True)
    classifier = pipeline(
        "zero-shot-classification", model="facebook/bart-large-mnli"
    )
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    print("models loaded")
    app.run()

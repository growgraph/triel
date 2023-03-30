import argparse
import os
import pkgutil
import unittest

import coreferee
import spacy
import yaml
from graph_cast.util import ResourceHandler, equals

from lm_service.linking import EntityLinkerManager
from lm_service.top import cast_response_to_unfolded, text_to_rel_graph


class TestREL(unittest.TestCase):
    cpath = os.path.dirname(os.path.realpath(__file__))

    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound_v2.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    conf = {
        "BERN_V2": {
            "url": "http://192.168.1.11:8888/plain",
            "text_field": "text",
            "threshold": 0.75,
        },
        "FISHING": {
            "url": "http://192.168.1.11:8090/service/disambiguate",
            "text_field": "text",
            "extra_args": {
                "language": {"lang": "en"},
                "mentions": ["ner", "wikipedia"],
            },
        },
    }

    conf_bern_external = {
        "BERN_V2": {
            "url": "http://bern2.korea.ac.kr/plain",
            "text_field": "text",
        }
    }

    def __init__(self, reset):
        super().__init__()
        self.reset = reset

    def test_iterate_linking_bern(self):
        text = "Diabetic ulcers are related to burns."
        # text = (
        #     "Thousands of exoplanets have been discovered by the end of the"
        #     " 2010s; some have minimum mass measurements from the radial"
        #     " velocity method while others that are seen to transit their"
        #     " parent stars have measures of their physical size."
        # )

        elm = EntityLinkerManager(self.conf_bern_external)

        response = text_to_rel_graph(text, self.nlp, self.rules, elm)
        response_jsonlike = cast_response_to_unfolded(
            response, cast_triple_version="v1"
        )

        if not self.reset:
            ref = ResourceHandler.load(
                "test.reference.el", "iterate_linking_bern.json"
            )
            self.assertEqual(response_jsonlike, ref)

        else:
            ResourceHandler.dump(
                response_jsonlike,
                os.path.join(
                    self.cpath, "reference/el/iterate_linking_bern.json"
                ),
            )

    def test_linking(self):
        text = "Diabetic ulcers are related to burns."
        # text = (
        #     "Thousands of exoplanets have been discovered by the end of the"
        #     " 2010s; some have minimum mass measurements from the radial"
        #     " velocity method while others that are seen to transit their"
        #     " parent stars have measures of their physical size."
        # )

        elm = EntityLinkerManager(self.conf)

        response = text_to_rel_graph(
            text, self.nlp, self.rules, elm, debug=True
        )
        response_jsonlike = cast_response_to_unfolded(
            response, cast_triple_version="v1"
        )

        if not self.reset:
            ref = ResourceHandler.load(
                "test.reference.el", f"iterate_linking_bern.json"
            )
            self.assertEqual(response_jsonlike, ref)

        else:
            ResourceHandler.dump(
                response_jsonlike,
                os.path.join(self.cpath, "reference/el/linking.json"),
            )

    def runTest(self):
        # self.test_iterate_linking_bern()
        self.test_linking()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset", action="store_true", help="reset test results"
    )
    args = parser.parse_args()
    suite = unittest.TestSuite()
    suite.addTest(TestREL(args.reset))
    unittest.TextTestRunner(verbosity=2).run(suite)

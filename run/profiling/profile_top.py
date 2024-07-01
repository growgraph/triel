# pylint: disable=E1101

import os
import pkgutil
from multiprocessing.managers import BaseManager
from pprint import pprint

import spacy
import yaml
from suthing import SProfiler

from lm_service.linking.onto import EntityLinkerManager
from lm_service.top import cast_response_to_unfolded, text_to_rel_graph


def main():
    os.path.dirname(os.path.realpath(__file__))

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound_v2.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

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

    text = "Diabetic ulcers are related to burns."
    # text = (
    #     "Thousands of exoplanets have been discovered by the end of the"
    #     " 2010s; some have minimum mass measurements from the radial"
    #     " velocity method while others that are seen to transit their"
    #     " parent stars have measures of their physical size."
    # )

    elm = EntityLinkerManager(conf)

    class LocalManager(BaseManager):
        pass

    LocalManager.register("SProfiler", SProfiler)

    with LocalManager() as manager:
        sp = manager.SProfiler()

        response = text_to_rel_graph(text, nlp, rules, elm, _profiler=sp)
        response_jsonlike = cast_response_to_unfolded(
            response, cast_triple_version="v1"
        )
        stats = sp.view_stats()
    print(response_jsonlike)
    pprint(stats)


if __name__ == "__main__":
    main()

import os
import pkgutil
from multiprocessing.managers import BaseManager
from pprint import pprint

import coreferee
import spacy
import yaml
from suthing import SProfiler

from lm_service.linking import EntityLinkerManager
from lm_service.top import cast_response_to_unfolded, text_to_rel_graph


def main():
    os.path.dirname(os.path.realpath(__file__))

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound_v2.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    conf = {
        "BERN_V2": {
            "url": "http://10.0.0.3:8888/plain",
            "text_field": "text",
            "threshold": 0.75,
        },
        "FISHING": {
            "url": "http://10.0.0.3:8090/service/disambiguate",
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

    def get_manager():
        m = LocalManager()
        m.start()
        return m

    LocalManager.register("SProfiler", SProfiler)

    manager = get_manager()
    sp = manager.SProfiler()

    response = text_to_rel_graph(text, nlp, rules, elm, _profiler=sp)
    response_jsonlike = cast_response_to_unfolded(
        response, cast_triple_version="v1"
    )
    print(response_jsonlike)
    pprint(sp.view_stats())


if __name__ == "__main__":
    main()

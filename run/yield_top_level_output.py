import argparse
import os
import pkgutil

import coreferee
import spacy
import yaml
from suthing import FileHandle, equals

from lm_service.linking import EntityLinkerFailed, EntityLinkerManager
from lm_service.top import cast_response_to_unfolded, text_to_rel_graph

cpath = os.path.dirname(os.path.realpath(__file__))

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

conf_bern_external = {
    "BERN_V2": {
        "url": "http://bern2.korea.ac.kr/plain",
        "text_field": "text",
    }
}


def run():
    text = "Diabetic ulcers are related to burns."
    text = (
        "The evolution and diversification of euthyneuran slugs and"
        " snails was likely strongly influenced by habitat transitions"
        " from marine to terrestrial and limnic systems."
        " Well-supported euthyneuran phylogenies with detailed"
        " morphological data can provide information on the"
        " historical, biological and ecological background in which"
        " these habitat shifts took place allowing for comparison"
        " across taxa. Acochlidian slugs are ‘basal pulmonates’ with"
        " uncertain relationships to other major panpulmonate clades."
        " They present a unique evolutionary history with"
        " representatives in the marine mesopsammon, but also benthic"
        " lineages in brackish water, limnic habitats and"
        " (semi-)terrestrial environments. We present the first"
        " comprehensive molecular phylogeny on Acochlidia, based on a"
        " global sampling that covers nearly 85 % of the described"
        " species diversity, and additionally, nearly doubles known"
        " diversity by undescribed taxa. Our phylogenetic hypotheses"
        " are highly congruent with previous morphological analyses"
        " and confirm all included recognized families and genera. We"
        " further establish an ancestral area chronogram for"
        " Acochlidia, document changes in diversification rates in"
        " their evolution via the birth-death-shift-model and"
        " reconstruct the ancestral states for major ecological"
        " traits. Based on our data, Acochlidia originated from a"
        " marine, mesopsammic ancestor adapted to tropical waters, in"
        " the mid Mesozoic Jurassic. We found that the two major"
        " subclades present a remarkably different evolutionary"
        " history. The Microhedylacea are morphologically"
        " highly-adapted to the marine mesopsammon. They show a"
        " circum-tropical distribution with several independent shifts"
        " to temperate and temperate cold-habitats, but remained in"
        " stunning morphological and ecological stasis since the late"
        " Mesozoic. Their evolutionary specialization, which includes"
        " a remarkable and potentially irreversible ‘meiofaunal"
        " syndrome’, guaranteed long-term survival and locally high"
        " species densities but also resembles a dead-end road to"
        " morphological and ecological diversity. In contrast, the"
        " Hedylopsacea are characterized by morphological flexibility"
        " and ecologically by independent habitat shifts out of the"
        " marine mesopsammon, conquering (semi-)terrestrial and limnic"
        " habitats. Originating from interstitial ancestors with"
        " moderate adaptations to the mesopsammic world, they"
        " reestablished a benthic lifestyle and secondary ‘gigantism’"
        " in body size. The major radiations and habitat shifts in"
        " hedylopsacean families occured in the central Indo-West"
        " Pacific in the Paleogene. In the Western Atlantic only one"
        " enigmatic representative is known probably presenting a"
        " relict of a former pan-Tethys distribution of the clade."
        " This study on acochlidian phylogeny and biogeography adds"
        " another facet of the yet complex panpulmonate evolution and"
        " shows the various parallel pathways in which these snails"
        " and slugs invaded non-marine habitats. Given the complex"
        " evolutionary history of Acochlidia, which represent only a"
        " small group of Panpulmonata, this study highlights the need"
        " to generate comprehensively-sampled species-level"
        " phylogenies to understand euthyneuran evolution."
    )

    # text = "what"
    # text = ""
    # text = "John "

    elm = EntityLinkerManager(conf)
    response = text_to_rel_graph(text, nlp, rules, elm)

    response_jsonlike = cast_response_to_unfolded(
        response, cast_triple_version="v1"
    )

    response_jsonlike["publication"] = {
        "arxiv": "current.123",
        "created": "2015-01-01",
    }
    FileHandle.dump(
        response_jsonlike,
        "/tmp/lm_top.json.gz",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    run()

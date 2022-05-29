import coreferee, spacy

nlp = spacy.load("en_core_web_trf")
nlp.add_pipe("coreferee")

# doc = nlp("CHEOPS is a European space telescope to determine the size of known extrasolar planets, which will allow the estimation of their mass, density, composition and their formation. It is the first Small-class mission in ESA's Cosmic Vision science programme.")
#
# doc._.coref_chains.print()


doc = nlp(
    "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space telescope to determine the size of known extrasolar planets, which will allow the estimation of their mass, density, composition and their formation.  It is the first Small-class mission in ESA's Cosmic Vision science programme launched on 18 December 2019"
)

doc._.coref_chains.print()

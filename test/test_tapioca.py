import spacy

nlp = spacy.blank("en")
nlp.add_pipe("opentapioca")
doc = nlp("Christian Drosten works in Germany.")
for span in doc.ents:
    print(
        (span.text, span.kb_id_, span.label_, span._.description, span._.score)
    )

doc = nlp("RNNs and CNNs are types of neural networks")
for span in doc.ents:
    print(
        (span.text, span.kb_id_, span.label_, span._.description, span._.score)
    )

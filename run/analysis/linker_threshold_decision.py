import pandas as pd
import requests
import spacy
import suthing

from lm_service.text import normalize_text

endpoint_url = "https://www.wikidata.org/w/api.php"


def get_wikidata_descriptions(entities):
    params = {
        "action": "wbgetentities",
        "ids": "|".join(entities),
        "format": "json",
        "props": "descriptions",
        "languages": "en",
    }
    response = requests.get(endpoint_url, params=params)
    data = response.json()
    descriptions = {}
    for entity, details in data["entities"].items():
        descriptions[entity] = (
            details.get("descriptions", {})
            .get("en", {})
            .get("value", "No description available")
        )
    return descriptions


def run():
    df = suthing.FileHandle.load(fpath="./debug/dump/cure51.all.el.v2.csv", index_col=0)
    texto = suthing.FileHandle.load(fpath="./debug/data/cure51.json")
    text = texto["text"]

    nlp = spacy.load("en_core_web_trf")
    phrases = normalize_text(text, nlp)
    text_used = " ".join(phrases)
    # sns.displot(df, hue="linker_type", x="score", kind="kde")

    df_sorted = df.sort_values(["linker_type", "score"])

    wiki_entities = df_sorted.loc[df_sorted["linker_type"] == "FISHING", "id"].to_list()

    n = 10
    batched = [wiki_entities[i : i + n] for i in range(0, len(wiki_entities), n)]

    acc = []
    for b in batched:
        descriptions = get_wikidata_descriptions(b)
        acc += [descriptions]

    df_fetched = pd.concat([pd.DataFrame(item.items()) for item in acc]).rename(
        columns={0: "id", 1: "desc"}
    )

    df_analysis = df_fetched.merge(
        df[["id", "score", "a", "b"]], on="id", how="left"
    ).sort_values("score")

    agg = []
    for ix, row in df_analysis.head(75).tail(10).iterrows():
        ii, desc, score, a, b = row
        agg += [(*row, text_used[a:b])]
        print(a, b)
        print(text_used[a:b], "|", desc, "|", score)

    _ = pd.DataFrame(agg, columns=["id", "desc", "score", "a", "b", "mention"])
    # df_sorted_a = df.sort_values(["a"])
    # df_sorted_a.loc[df_sorted_a["a"] >= 377].head(20)

    # 0.55 for FISHING


if __name__ == "__main__":
    run()

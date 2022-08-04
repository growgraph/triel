import re

from unidecode import unidecode


def split_tokens_into_phrases(tokens_list, terminal_puncts=(".", "!", "?")):
    phrases = []
    cur_phrase = [tokens_list[0]]

    # split only if a white space follows and the following letter is capital
    for token0, token1, token2 in zip(
        tokens_list[:-2], tokens_list[1:-1], tokens_list[2:]
    ):
        cur_phrase.append(token1)
        if (
            not token0[-1].isupper()
            and token1 in terminal_puncts
            and token2[0].isupper()
        ):
            phrases.append(cur_phrase)
            cur_phrase = []
    cur_phrase.append(tokens_list[-1])
    phrases.append(cur_phrase)
    return phrases


def normalize_input_text(article, terminal_full_stop=True):
    article = unidecode(article)

    tokenized_agg = re.findall(r"[\w']+|[.,!?;:-\\(\\)]", article)

    phrases = split_tokens_into_phrases(tokenized_agg)
    if not terminal_full_stop:
        phrases = [ph[:-1] for ph in phrases]
    phrases = [" ".join(ph) for ph in phrases]
    return phrases

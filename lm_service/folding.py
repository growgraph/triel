import logging

logger = logging.getLogger(__name__)


def get_flag(props, rules):
    hows = {"eq"}
    for r in rules:
        for sr in r:
            if "how" in sr:
                hows |= {sr["how"]}

    conclusion = []
    for r in rules:
        flag = []
        for subrule in r:
            if "how" not in subrule:
                how = "__eq__"
            else:
                how = subrule["how"]
            foo = get_foo(how, props[subrule["key"]])
            subflag = foo(subrule["value"])
            flag.append(subflag)
        conclusion += [all(flag)]
    return any(conclusion)


def get_foo(how: str, obj):
    try:
        foo = getattr(obj, how)
        return foo
    except:
        logger.info(f"bare did not fly {how}")
    try:
        how_builtins = f"__{how}__"
        foo = getattr(obj, how_builtins)
        return foo
    except Exception as e:
        raise e

def get_flag(props, rules):
    conclusion = []
    for r in rules:
        flag = []
        for subrule in r:
            if "how" not in subrule:
                flag.append(props[subrule["key"]] == subrule["value"])
            elif subrule["how"] == "contains":
                flag.append(subrule["value"] in props[subrule["key"]])
        conclusion += [all(flag)]
    return any(conclusion)

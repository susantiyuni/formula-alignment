# representations/semantic.py


# =========================================================
# Row → semantic text helpers
# =========================================================
def get_descriptions(row):
    """
    Collects all semantic descriptions for one formula row.
    """
    texts = []

    for lbl, desc in zip(row.get("conceptLabels", []),
                         row.get("conceptDescriptions", [])):
        if lbl:
            texts.append(lbl.strip())
        if desc:
            texts.append(desc.strip())

    if row.get("itemDescription"):
        texts.append(row["itemDescription"].strip())

    if row.get("itemLabel"):
        texts.append(row["itemLabel"].strip())

    return list(set(texts))


def get_symbol_description(row, label):
    """
    Returns description text for one symbol label only.
    Used for node-level semantic hints.
    """
    descs = []
    for lbl, desc in zip(row.get("conceptLabels", []),
                         row.get("conceptDescriptions", [])):
        if lbl == label and desc:
            descs.append(desc.strip())

    return " ".join(descs) if descs else ""


# =========================================================
# Whole dataset → semantic texts
# =========================================================
def build_semantic_texts(data):
    """
    Returns list[str] aligned with dataset rows.
    Each string = concatenated semantic info.
    """
    return [" ".join(get_descriptions(r)) for r in data]

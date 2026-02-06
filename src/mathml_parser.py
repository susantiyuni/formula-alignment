# mathml/parser.py

from lxml import etree

MATHML_NS = "{http://www.w3.org/1998/Math/MathML}"


# =========================================================
# Parse MathML string → <math> root node
# =========================================================
def parse_mathml_tree(mathml_str: str):
    """
    Returns the <math> root element from a MathML string.
    Handles both pure MathML and embedded HTML.
    """
    try:
        root = etree.fromstring(mathml_str.encode("utf8"))
        if "math" in etree.QName(root).localname.lower():
            return root
    except Exception:
        pass

    parser = etree.HTMLParser()
    tree = etree.fromstring(mathml_str.encode("utf8"), parser=parser)
    math_node = tree.find(".//math")

    if math_node is None:
        raise ValueError("No <math> node found")

    return math_node


# =========================================================
# MathML → OPT tree
# =========================================================
def mathml_to_opt(node):
    """
    Converts MathML XML → OPT tree.

    OPT format:
    {
        "type": "sym" | "op" | "func",
        "value": str,
        "children": [...]
    }
    """
    tag = etree.QName(node.tag).localname
    children = [c for c in node if isinstance(c.tag, str)]

    # ---------------------
    # Leaf nodes
    # ---------------------
    if tag in ("mi", "mn", "mtext"):
        return {"type": "sym", "value": (node.text or "").strip(), "children": []}

    if tag == "mo":
        return {"type": "op", "value": (node.text or "").strip(), "children": []}

    # ---------------------
    # Structured operators
    # ---------------------
    def two(name):
        return {
            "type": "func",
            "value": name,
            "children": [mathml_to_opt(children[0]), mathml_to_opt(children[1])]
        }

    if tag == "msup" and len(children) >= 2:
        return two("power")

    if tag == "msub" and len(children) >= 2:
        return two("subscript")

    if tag == "mfrac" and len(children) >= 2:
        return two("frac")

    if tag == "mroot" and len(children) >= 2:
        return two("root")

    if tag == "msqrt" and len(children) >= 1:
        return {
            "type": "func",
            "value": "sqrt",
            "children": [mathml_to_opt(children[0])]
        }

    if tag == "msubsup" and len(children) >= 3:
        return {
            "type": "func",
            "value": "subsup",
            "children": [
                mathml_to_opt(children[0]),
                mathml_to_opt(children[1]),
                mathml_to_opt(children[2]),
            ]
        }

    if tag == "mrow":
        return {
            "type": "func",
            "value": "group",
            "children": [mathml_to_opt(c) for c in children]
        }

    # skip wrappers
    if tag == "semantics" and children:
        return mathml_to_opt(children[0])

    if tag in ("annotation", "annotation-xml"):
        return {"type": "func", "value": "ignored", "children": []}

    if tag == "mstyle":
        return {
            "type": "func",
            "value": "style_group",
            "children": [mathml_to_opt(c) for c in children]
        }

    # fallback
    return {
        "type": "func",
        "value": tag,
        "children": [mathml_to_opt(c) for c in children]
    }

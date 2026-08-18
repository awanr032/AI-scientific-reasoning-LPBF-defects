from lpbf_defect_reasoning.text import normalize_query


def test_normalize_query_lowercases():
    assert normalize_query("LASER Power") == "laser power"


def test_normalize_query_rewrites_known_phrases():
    q = normalize_query("Why does keyhole porosity occur with unstable melt pool?")
    assert "keyhole_porosity" not in q  # "keyhole porosity" maps to "keyhole", not itself
    assert "keyhole" in q
    assert "melt_pool_instability" in q


def test_normalize_query_leaves_unknown_phrases_untouched():
    assert normalize_query("hatch spacing") == "hatch spacing"

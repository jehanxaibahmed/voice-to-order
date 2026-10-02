from voice_to_order.text import Op, align, normalise_tokens, word_edit_distance


def test_normalise_strips_punctuation_but_keeps_codes() -> None:
    assert normalise_tokens("Hi, it's Sam — order OM-12, please!") == [
        "hi", "it's", "sam", "order", "om-12", "please",
    ]  # fmt: skip


def test_align_reports_each_edit_type() -> None:
    pairs = align(["a", "b", "c"], ["a", "x", "c", "d"])
    assert [p.op for p in pairs] == [Op.MATCH, Op.SUBSTITUTE, Op.MATCH, Op.INSERT]
    assert word_edit_distance(["a", "b", "c"], ["a", "c"]) == 1


def test_align_empty_inputs() -> None:
    assert align([], []) == []
    assert [p.op for p in align(["a"], [])] == [Op.DELETE]
    assert [p.op for p in align([], ["a"])] == [Op.INSERT]

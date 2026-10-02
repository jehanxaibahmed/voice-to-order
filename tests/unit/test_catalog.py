from pathlib import Path

import pytest

from voice_to_order.extraction import Catalog


@pytest.fixture
def catalog(samples_dir: Path) -> Catalog:
    return Catalog.load(samples_dir / "catalog.json")


@pytest.mark.parametrize(
    ("spoken", "code", "sku"),
    [
        ("oat milk", None, "OM-12"),
        ("barista oat milk", None, "OM-12"),
        ("anything", "om-12", "OM-12"),
        ("croissants", None, "CR-24"),
        ("free range eggs", None, "EG-30"),
        ("takeaway cups", None, "TC-80"),
        ("semi skimmed milk", None, "SM-02"),
        ("crates of whole milk", None, "WM-01"),
    ],
)
def test_matches_spoken_names(catalog: Catalog, spoken: str, code: str | None, sku: str) -> None:
    match = catalog.match(spoken, code)
    assert match is not None
    assert match.product.sku == sku


def test_unknown_product_has_no_match(catalog: Catalog) -> None:
    assert catalog.match("smoked salmon") is None
    assert catalog.match("") is None


def test_unknown_code_falls_back_to_description(catalog: Catalog) -> None:
    match = catalog.match("sourdough", "XX-99")
    assert match is not None and match.product.sku == "SB-10"


def test_vocabulary_lists_codes_and_names(catalog: Catalog) -> None:
    assert "OM-12 Barista oat milk 1L" in catalog.vocabulary()


def test_keyterms_are_short_and_unique(catalog: Catalog) -> None:
    terms = catalog.keyterms()
    assert {"OM-12", "Barista oat milk 1L", "oat milk"} <= set(terms)
    assert len(terms) == len(set(terms))

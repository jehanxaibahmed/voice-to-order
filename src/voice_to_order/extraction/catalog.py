"""Product catalogue and fuzzy matching of spoken product names to SKUs."""

import json
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import BaseModel, Field

from voice_to_order.text import normalise_tokens

DEFAULT_MATCH_THRESHOLD = 0.75


class Product(BaseModel):
    sku: str
    name: str
    unit: str
    aliases: list[str] = Field(default_factory=list)


class CatalogMatch(BaseModel):
    product: Product
    score: float


def _norm(text: str) -> str:
    return " ".join(normalise_tokens(text))


class Catalog:
    def __init__(self, products: list[Product], *, threshold: float = DEFAULT_MATCH_THRESHOLD):
        self.products = products
        self._threshold = threshold
        self._by_sku = {_norm(p.sku): p for p in products}

    @classmethod
    def load(cls, path: Path) -> "Catalog":
        data = json.loads(path.read_text())
        return cls([Product.model_validate(p) for p in data["products"]])

    def vocabulary(self) -> list[str]:
        """Product names and codes, used to bias transcription and prompts."""
        return [f"{p.sku} {p.name}" for p in self.products]

    def keyterms(self) -> list[str]:
        """Short terms to boost in speech recognition: SKUs, product names and aliases."""
        terms: dict[str, None] = {}
        for product in self.products:
            for term in (product.sku, product.name, *product.aliases):
                terms.setdefault(term, None)
        return list(terms)

    def match(self, description: str, product_code: str | None = None) -> CatalogMatch | None:
        if product_code and (product := self._by_sku.get(_norm(product_code))):
            return CatalogMatch(product=product, score=1.0)

        spoken = _norm(description)
        if not spoken:
            return None
        best: CatalogMatch | None = None
        for product in self.products:
            for candidate in (product.name, *product.aliases):
                score = self._similarity(spoken, _norm(candidate))
                if best is None or score > best.score:
                    best = CatalogMatch(product=product, score=round(score, 3))
        return best if best and best.score >= self._threshold else None

    @staticmethod
    def _similarity(spoken: str, candidate: str) -> float:
        # A spoken phrase that contains the full candidate ("two crates of whole milk please")
        # is a strong match even when the strings differ in length.
        if f" {candidate} " in f" {spoken} ":
            return 0.95 if spoken != candidate else 1.0
        return SequenceMatcher(None, spoken, candidate).ratio()

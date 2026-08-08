"""The product-defining invariant (spec §0, §21): no aggregate score, no
ranking, no certification — enforced against both the schemas and the source.
"""

import re
from pathlib import Path

from sieve.core import models

SRC = Path(__file__).resolve().parents[2] / "src"

FORBIDDEN_FIELD_PARTS = ("score", "rating", "ranking", "grade", "stars",
                         "percentile", "leaderboard")
# "rank" alone would false-positive on legitimate statistics vocabulary
# (rank-based tests); the parts above are the product-scope terms.

FORBIDDEN_TEXT = (
    re.compile(r"reality\s+score", re.I),
    re.compile(r"overall\s+score", re.I),
    re.compile(r"\bcertified\b", re.I),
    re.compile(r"\bleaderboard\b", re.I),
)


def _all_models():
    import pydantic
    for name in dir(models):
        obj = getattr(models, name)
        if isinstance(obj, type) and issubclass(obj, pydantic.BaseModel):
            yield name, obj


def test_no_schema_field_aggregates():
    for name, cls in _all_models():
        for field in cls.model_fields:
            for bad in FORBIDDEN_FIELD_PARTS:
                assert bad not in field.lower(), f"{name}.{field}"


def test_no_forbidden_language_in_source():
    files = list(SRC.rglob("*.py")) + list(SRC.rglob("*.j2"))
    assert files
    for f in files:
        text = f.read_text()
        for pat in FORBIDDEN_TEXT:
            assert not pat.search(text), f"{f}: {pat.pattern}"


def test_profile_offers_no_aggregation():
    from sieve.core.models import ValidationProfile
    profile = ValidationProfile.not_tested()
    public = [a for a in dir(profile)
              if not a.startswith(("_", "model_")) and a not in
              ("dimensions", "as_dict", "not_tested")]
    # pydantic adds construct/copy/dict/json/... helpers prefixed model_ or
    # deprecated aliases; anything else numeric-sounding would be a smell.
    for attr in public:
        assert attr in ("construct", "copy", "dict", "json", "parse_file",
                        "parse_obj", "parse_raw", "schema", "schema_json",
                        "update_forward_refs", "validate", "from_orm"), attr

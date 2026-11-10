from pathlib import Path
import pytest
from src.knowledge import list_concepts, read_concept, search_concepts


def test_bundle_has_typed_concepts():
    docs = list_concepts()
    assert len(docs) >= 5
    assert all(doc['type'] and doc['id'].endswith('.md') for doc in docs)


def test_search_family_sources():
    result = search_concepts('SUV para família, bagagem e segurança')
    assert result
    assert any(doc['id'] == 'rules/suv-familiar.md' for doc in result)
    assert all(doc['source'].startswith('okf:') for doc in result)


def test_reject_path_outside_bundle(tmp_path: Path):
    external = tmp_path / 'fake.md'
    external.write_text('---\ntype: Test\n---\ntest', encoding='utf-8')
    with pytest.raises(ValueError):
        read_concept(external)


def test_missing_type_is_invalid(tmp_path: Path):
    doc = tmp_path / 'invalid.md'
    doc.write_text('---\ntitle: Invalid\n---\nContent', encoding='utf-8')
    with pytest.raises(ValueError, match='type'):
        read_concept(doc, tmp_path)

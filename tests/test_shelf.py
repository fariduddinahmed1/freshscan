"""ShelfStore tests on tmp_path sqlite. Real sqlite, no mocks."""
from core.shelf import ShelfStore


def _store(tmp_path):
    return ShelfStore(str(tmp_path / "shelf.db"))


def test_add_get_all_roundtrip(tmp_path):
    store = _store(tmp_path)
    store.add("1", "A", "Apple", "a@b.com")
    items = store.get_all()
    assert items["1_A"]["item_name"] == "Apple"
    assert items["1_A"]["email"] == "a@b.com"
    assert items["1_A"]["scans"] == []


def test_record_scan_appends_and_sets_latest(tmp_path):
    store = _store(tmp_path)
    store.add("1", "A", "Apple", None)
    store.record_scan("1", "A", {"category": "Fresh", "confidence": 99.0, "shelf_life": "6 days"})
    item = store.get_all()["1_A"]
    assert len(item["scans"]) == 1
    assert item["scans"][0]["category"] == "Fresh"
    assert item["latest"]["category"] == "Fresh"
    assert item["latest"]["confidence"] == 99.0


def test_readd_resets_latest(tmp_path):
    store = _store(tmp_path)
    store.add("1", "A", "Apple", None)
    store.record_scan("1", "A", {"category": "Fresh", "confidence": 99.0, "shelf_life": "6 days"})
    store.add("1", "A", "Apple", None)
    assert "latest" not in store.get_all()["1_A"]


def test_delete_removes_item_and_scans(tmp_path):
    store = _store(tmp_path)
    store.add("1", "A", "Apple", None)
    store.record_scan("1", "A", {"category": "Fresh", "confidence": 99.0, "shelf_life": "6 days"})
    assert store.delete("1", "A") is True
    assert store.get_all() == {}

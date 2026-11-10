from src.vehicle_service import VehicleService, public_vehicle


class FakeSearch:
    def __init__(self):
        self.calls = []

    def bm25(self, question, filters, size):
        self.calls.append(('bm25', filters))
        return [{'id': 'V001', 'preco': 89000, 'marca': 'fiat', 'carroceria': 'suv', 'cambio': 'automatico', 'vetor': [0.1, 0.2]},
                {'id': 'V002', 'preco': 112000}]

    def vector(self, question, filters, size):
        self.calls.append(('vector', filters))
        return [{'id': 'V001', 'preco': 89000, 'marca': 'fiat', 'carroceria': 'suv', 'cambio': 'automatico'}, {'id': 'V003', 'preco': 90000, 'carroceria': 'suv', 'cambio': 'automatico'}]


class FakeClient:
    def get(self, *, index, id, _source_excludes):
        return {'_source': {'id': id, 'marca': 'fiat', 'preco': 89000, 'vetor': [1]}}


def test_search_hybrid_invokes_both_engines():
    search = FakeSearch()
    out = VehicleService(search, FakeClient()).search_vehicles('SUV até 100 mil', 'hybrid', 3)
    assert [c[0] for c in search.calls] == ['bm25', 'vector']
    assert out['filters']['preco_max'] == 100000
    assert out['vehicles'][0]['id'] == 'V001'
    assert all('vetor' not in v for v in out['vehicles'])
    assert all(v['preco'] <= 100000 for v in out['vehicles'])


def test_only_bm25_mode():
    search = FakeSearch()
    VehicleService(search, FakeClient()).search_vehicles('Corolla', 'bm25', 2)
    assert [c[0] for c in search.calls] == ['bm25']


def test_details_do_not_expose_embeddings():
    out = VehicleService(FakeSearch(), FakeClient()).details('V001')
    assert out['source'] == 'catalog:V001'
    assert 'vetor' not in out['vehicle']


def test_reject_bad_id():
    import pytest
    with pytest.raises(ValueError):
        VehicleService(FakeSearch(), FakeClient()).details('../../etc/passwd')


def test_public_vehicle_allowlist():
    assert public_vehicle({'id': 'V001', 'vetor': [1], 'secret': 'x'}) == {'id': 'V001'}

from src.planning import plan, requires_compare
from src.mcp_client import normalize_tool_names, parse_tool_result


def test_comparison_requested():
    assert requires_compare('Compare três SUVs para a família')
    assert 'comparar_veiculos' in plan('Compare três SUVs')['plano']


def test_comparison_not_requested():
    assert not requires_compare('Quero um Corolla')
    assert 'comparar_veiculos' not in plan('Quero um Corolla')['plano']


def test_adapter_prefix_normalization():
    class FakeTool:
        def __init__(self, name):
            self.name = name

    t1, t2 = FakeTool('veiculos_buscar_veiculos'), FakeTool('conhecimento_buscar_conhecimento')
    assert normalize_tool_names([t1, t2]) == {'buscar_veiculos': t1, 'buscar_conhecimento': t2}


def test_mcp_text_json():
    assert parse_tool_result([{'type': 'text', 'text': '{"vehicles": []}'}]) == {'vehicles': []}

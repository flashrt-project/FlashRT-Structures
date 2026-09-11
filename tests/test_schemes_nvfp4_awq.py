"""The NVFP4-AWQ scheme decides from the recipe payload (moved from
FlashRT's test_core_quantization when the structures layer split out)."""
import pytest
import torch


def test_nvfp4_awq_scheme_decides_with_the_recipe_payload():
    import flashrt_structures.schemes as schemes
    from flashrt_structures.schemes import Nvfp4Awq, PointStat, \
        validate_request

    scheme = schemes.get("nvfp4_awq")
    assert isinstance(scheme, Nvfp4Awq)

    class _Pt:
        def __init__(self, path, name):
            self.path, self.name = path, name

    req = scheme.statistics([_Pt("a.mlp", "x_after_norm"),
                             _Pt("a.mlp.down_proj", "act_after_mul")])
    assert all(ps == PointStat("amax", "channel") for ps in req.values())
    validate_request(req)          # the collector measures this today

    report = {
        "layers.0.mlp": {"layers.0.mlp.down_proj|act_after_mul": None},
        "layers.0.self_attn": {"layers.0.self_attn|x": None},
    }
    d = scheme.decide(report)
    assert d.formats == {"layers.0.mlp": "nvfp4_awq"}
    assert d.params["layers.0.mlp"] == {
        "alpha": 0.5, "clamp": [0.25, 4.0], "recipe": "balance"}
    assert d.keep_host == ("layers.0.self_attn",)

    with pytest.raises(ValueError, match="recipe"):
        Nvfp4Awq(recipe="magic")

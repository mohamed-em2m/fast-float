"""Registry matrix: the single source of truth must stay self-consistent."""

from fast_float.onnx.backends import COMPONENTS, HOT_COMPONENTS
from fast_float.onnx.registry import (
    constructor,
    get,
    hot,
    names,
    post_load,
    replicable,
    validate,
)


def test_validate_is_clean():
    assert validate() == []


def test_names_match_backends_components():
    assert set(names()) == set(COMPONENTS)


def test_stages_partition_the_components():
    assert set(post_load()) | set(constructor()) == set(names())
    assert set(post_load()) & set(constructor()) == set()


def test_hot_matches_backends_hot_components():
    assert set(hot()) == set(HOT_COMPONENTS)


def test_replicable_is_decoder_only_and_post_load():
    assert replicable() == ("decoder",)
    for name in replicable():
        assert get(name).backend_stage == "post_load"


def test_fmt_is_sticky_and_others_transient():
    assert get("fmt").degrade_on_error == "sticky"
    assert get("direction").degrade_on_error == "transient"
    assert get("decoder").degrade_on_error == "transient"


def test_decoder_carries_the_relaxed_parity_gate():
    dec = get("decoder")
    assert dec.replicable is True
    assert dec.min_cosine == 0.995
    assert dec.max_mean == 0.05


def test_every_component_has_a_default_onnx_path():
    for name in names():
        assert get(name).onnx_default

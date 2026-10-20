"""Spec / fingerprint / cache: the pure-CPU core of the builder layer."""

import pytest

from fast_float.builders.spec import ArtifactCache, BuildSpec, RunSpec, build_fingerprint


def test_buildspec_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown backend"):
        BuildSpec(component="direction", backend="nope")


def test_buildspec_rejects_unknown_precision():
    with pytest.raises(ValueError, match="unknown precision"):
        BuildSpec(component="direction", backend="onnx", precision="bf16")


def test_runspec_rejects_unknown_degrade_mode():
    with pytest.raises(ValueError, match="unknown degrade_on_error"):
        RunSpec(component="fmt", degrade_on_error="nope")


def test_fingerprint_is_stable_and_env_sensitive():
    spec = BuildSpec(component="direction", backend="onnx", input_names=("x",), opt_shapes={"x": (1, 20)})
    env = {"torch": "2.0.1", "ort": "1.23.2", "cuda_ep": True}
    assert build_fingerprint(spec, env) == build_fingerprint(spec, dict(env))

    other = {**env, "cuda_ep": False}
    assert build_fingerprint(spec, env) != build_fingerprint(spec, other)

    # env keys outside the fingerprint set are ignored (e.g. gpu name).
    assert build_fingerprint(spec, env) == build_fingerprint(spec, {**env, "gpu": "T4"})


def test_fingerprint_changes_with_shapes_and_module_class():
    a = BuildSpec(component="fmt", backend="onnx", opt_shapes={"x": (1, 50, 512)})
    b = BuildSpec(component="fmt", backend="onnx", opt_shapes={"x": (2, 50, 512)})
    assert build_fingerprint(a) != build_fingerprint(b)
    assert build_fingerprint(a) != build_fingerprint(a, module_class="Other")


def test_artifact_cache_roundtrip(tmp_path):
    path = tmp_path / "manifest.json"
    cache = ArtifactCache(path)
    assert cache.get("missing") is None
    assert len(cache) == 0

    cache.put("abc", {"component": "direction", "path": "d.onnx"})
    assert "abc" in cache
    assert len(cache) == 1

    reloaded = ArtifactCache(path)
    assert reloaded.get("abc") == {"component": "direction", "path": "d.onnx"}


def test_artifact_cache_tolerates_missing_and_corrupt_file(tmp_path):
    assert len(ArtifactCache(tmp_path / "absent.json")) == 0
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert len(ArtifactCache(bad)) == 0

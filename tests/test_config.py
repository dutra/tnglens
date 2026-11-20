import pytest

from tnglens import config


def test_defaults_are_the_fiducial_settings():
    cfg = config.validate({"halos": [{"zoom": 14, "snapshot": 73}]})
    p = cfg["projection"]
    assert p["neighbors"] == 64 and p["support_floor_pkpc"] == 1.48
    assert p["core_width_arcsec"] == 400 and p["core_resolution"] == 4096
    assert p["environment_width_r200"] == 4 and p["environment_resolution"] == 2048
    assert p["depth_half_r200"] == 2 and p["target_cells_per_h"] == 8
    assert cfg["cosmology"] == "native" and cfg["particle_types"] == [0, 1, 2, 4, 5]
    assert "catalog" not in cfg
    (req,) = config.requests(cfg)
    assert req == config.Request(73, 14, None, ("xy", "yz", "zx"))


def test_lists_expand_and_duplicates_merge_axes():
    cfg = config.validate({"halos": [{"zoom": [1, 2], "snapshot": [72, 73], "axes": ["yz"]},
                                     {"zoom": 1, "snapshot": 73, "axes": ["xy", "yz"]},
                                     {"halo_id": 2571291, "snapshot": 73, "axes": "zx"}]})
    reqs = {(r.snapshot, r.zoom, r.halo_id): r.axes for r in config.requests(cfg)}
    assert len(reqs) == 5
    assert reqs[(73, 1, None)] == ("yz", "xy")
    assert reqs[(73, None, 2571291)] == ("zx",)


def test_cosmology_override_and_catalog_passthrough(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("cosmology: {H0: 70, Om0: 0.3}\n"
                    "halos: [{zoom: 14, snapshot: 73, axes: [yz]}]\n"
                    "catalog: {seed: 3, noise: {sigma_pos: 0.05}}\n"
                    "output_root: rel/out\n")
    cfg = config.load(path)
    assert cfg["cosmology"] == {"H0": 70.0, "Om0": 0.3}
    assert cfg["catalog"] == {"seed": 3, "noise": {"sigma_pos": 0.05}}  # passed unchanged
    assert cfg["output_root"].endswith("rel/out") and cfg["output_root"].startswith("/")


@pytest.mark.parametrize("bad", [
    {"projection": {"core_resolution": 1000}},
    {"projection": {"nieghbors": 64}},
    {"particle_types": [3]},
    {"cosmology": {"H0": 70}},
    {"source_redshifts": []},
    {"halos": [{"zoom": 1}]},
    {"halos": [{"zoom": 1, "halo_id": 3, "snapshot": 73}]},
    {"halos": [{"zoom": 1, "snapshot": 73, "axes": ["xz"]}]},
    {"workers": 0},
    {"workers": True},
    {"threads": 2.0},
    {"projection": {"neighbors": True}},
    {"projection": {"target_cells_per_h": True}},
    {"projection": {"core_width_arcsec": False}},
    {"particle_types": [True, 0]},
    {"source_redshifts": [True]},
    {"cosmology": {"H0": True, "Om0": 0.3}},
    {"halos": [{"zoom": True, "snapshot": 73}]},
    {"catalog": {"n_sytems": 3}},
    {"catalog": {"noise": {"sigma": 0.1}}},
    {"catalog": [1, 2]},
    {"simulation": {"name": "tng300"}},
])
def test_invalid_configs_are_rejected(bad):
    with pytest.raises(config.ConfigError):
        config.validate(bad)


def test_example_config_is_valid():
    from pathlib import Path
    example = Path(__file__).resolve().parents[1] / "configs" / "example.yaml"
    cfg = config.load(example)
    assert cfg["halos"]


def test_catalog_typo_fails_at_load_with_a_clear_message(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("halos: [{zoom: 14, snapshot: 73}]\ncatalog: {n_sytems: 3}\n")
    with pytest.raises(config.ConfigError, match="catalog: unknown catalog key 'n_sytems'"):
        config.load(path)


def test_unreadable_or_invalid_yaml_is_a_config_error(tmp_path):
    with pytest.raises(config.ConfigError, match="cannot read config"):
        config.load(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("halos: [unclosed\n")
    with pytest.raises(config.ConfigError, match="invalid YAML"):
        config.load(bad)

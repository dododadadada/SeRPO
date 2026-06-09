from grpo_online.config import OnlineConfig, load_config

def test_defaults():
    c = OnlineConfig()
    assert c.M == 6 and c.G == 8 and c.K == 1
    assert c.lr == 1e-6 and c.kl_beta == 0.04
    assert c.gen_ports == [8101]

def test_yaml_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("M: 4\nK: 2\ngen_ports: [8101, 8102]\n")
    c = load_config(str(p))
    assert c.M == 4 and c.K == 2 and c.gen_ports == [8101, 8102]

def test_outcome_type_field(tmp_path):
    # default is "continuous"
    assert OnlineConfig().outcome_type == "continuous"
    # and it round-trips through the strict YAML loader (would raise if unknown)
    p = tmp_path / "c.yaml"
    p.write_text("outcome_type: binary\n")
    c = load_config(str(p))
    assert c.outcome_type == "binary"

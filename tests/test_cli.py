import json

from pair_agent.cli import main


def test_cli_synthetic_report(tmp_path, capsys):
    assert main(["research", "--ticker", "KO", "--synthetic", "--end", "2026-10-19",
                 "--data-dir", str(tmp_path)]) == 0
    assert "ARTIFICIAL SYNTHETIC" in capsys.readouterr().out
    reports = list((tmp_path / "processed").glob("research-*.json"))
    assert json.loads(reports[0].read_text())["selected_pairs"][0]["ticker_a"] == "KO"

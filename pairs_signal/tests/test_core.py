import csv
import io

import pandas as pd

from pairs_signal import tasks
from pairs_signal.build_cases import event_days
from pairs_signal.label_headlines import rule_label
from pairs_signal.news import anonymize
from pairs_signal.runpod_deploy import env_flag


def test_anonymize_names_tickers_and_dates():
    m = {"TGT": "Company B", "WMT": "Company A"}
    assert anonymize("UBS Raises Price Target on Target (NYSE:TGT) to $88", m) == "UBS Raises Price Target on Company B to $88"
    assert anonymize("Walmart's Q3 2019 earnings beat; Target shares fall Nov. 5", m) == "Company A's Q3 [year] earnings beat; Company B shares fall [date]"
    assert anonymize("Spending Is Up (MA, V, MCD)", {"V": "Company A", "MA": "Company B"}) == "Spending Is Up (Company B, Company A, MCD)"
    assert anonymize("Cramer likes The Home Depot (HD) over Lowe's", {"HD": "Company A", "LOW": "Company B"}) == "Cramer likes The Company A over Company B"
    assert anonymize("Pete Najarian Thinks Pepsico Inc (PEP) Is Going Forward", {"KO": "Company A", "PEP": "Company B"}) == "Pete Najarian Thinks Company B Is Going Forward"


def test_one_excursion_is_one_event():
    # Crosses 2 three times within one excursion that never gets back under 0.5.
    z = pd.Series([0.0, 2.1, 1.9, 2.2, 1.8, 2.3, 0.4, 2.5], index=pd.RangeIndex(8))
    assert event_days(z) == [1, 7]


def test_event_rearms_after_horizon():
    z = pd.Series([0.0, 2.1] + [1.5] * 10 + [2.2], index=pd.RangeIndex(13))
    assert event_days(z) == [1, 12]


def test_parse_labels():
    t = tasks.headline_task()
    assert t.parse("m_and_a") == "m_and_a"
    assert t.parse('{"label": "noise"}') == "noise"
    assert t.parse("<think>maybe noise</think>\nanalyst_rating") == "analyst_rating"
    c = tasks.case_task()
    assert c.parse("no_revert") == "no_revert"
    assert c.parse("reverts") == "reverts"


def test_rule_labels():
    assert rule_label("Company A Names New CEO") == "management_change"
    assert rule_label("Company B to Acquire Rival for $2B") == "m_and_a"
    assert rule_label("Barclays Downgrades Company A to Underweight") == "analyst_rating"
    assert rule_label("Stocks That Hit 52-Week Highs On Friday") == "noise"
    assert rule_label("4 Stocks To Buy In The Wake Of The Trade War") != "m_and_a"


def test_runpod_env_flag_survives_csv_parsing():
    value = '[{"name":"model","path":"user/adapter"}]'
    flag = env_flag("LORA_MODULES", value)
    assert next(csv.reader(io.StringIO(flag))) == [f"LORA_MODULES={value}"]
    assert env_flag("ENABLE_LORA", "true") == "ENABLE_LORA=true"


def test_news_mapping_covers_every_source_label():
    import json

    from pairs_signal import config
    from pairs_signal.news_datasets import SOURCE_NAMES

    for task in ("news_event", "news_sentiment"):
        spec = json.loads((config.TASKS / f"{task}_labels.json").read_text())
        mapping = spec["source_mapping"]["map"]
        assert set(SOURCE_NAMES[task]) == set(mapping)
        assert {v for v in mapping.values() if v} == set(spec["labels"])


def test_news_tasks_parse_their_labels():
    event = tasks.get("news_event")
    assert event.parse("management_change") == "management_change"
    assert event.parse('{"label": "m_and_a"}') == "m_and_a"
    assert tasks.get("news_sentiment").parse("negative") == "negative"

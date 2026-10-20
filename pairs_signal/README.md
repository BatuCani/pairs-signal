# pairs-signal: Aktien-News-Klassifikator

Hackathon 2026-10-19 (distil labs, Langfuse, Runpod).
Produkt: ein kleines, lokal lauffaehiges Modell, das Aktien-Nachrichten einordnet.
Eingabe ist eine Schlagzeile oder ein News-Tweet zu einer Aktie, Ausgabe ist der Ereignistyp (10 Klassen) und optional die Stimmung (positiv, negativ, neutral).
Trainiert wird mit distil labs, serviert auf Runpod (vLLM mit LoRA), jeder Aufruf wird in Langfuse getract und die Fehler fliessen als neue Trainingsbeispiele zurueck.

Dieses Projekt liegt im Unterordner `pairs_signal/` des Repos und ist ein eigenes uv-Projekt; alle Befehle und Pfade unten gelten in diesem Ordner (`cd pairs_signal`).
Der Root des Repos beschreibt das Gesamtprojekt News Gate (`README.md`), daneben liegen `pitch/`, `ui/` und `training/`.

Alles laeuft ohne Keys mit `--dry-run`.
Keys kommen nur aus der Umgebung (`.env.example` zeigt alle Variablen).

## Datensaetze

Geprueft am 2026-10-19 (Seite bzw. API selbst geoeffnet, Lizenz gelesen, Download gemessen).

| Quelle | Lizenz | Zeilen | Label-Herkunft | Verwendung |
|---|---|---|---|---|
| [zeroshot/twitter-financial-news-topic](https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic) | MIT | 21.107 (16.990 train, 4.117 validation), 2,4 MB, < 1 s | vom Datensatz-Autor annotiert, 20 Themen | **news_event**, committet |
| [zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | MIT | 11.932 (9.543 / 2.388) | vom Datensatz-Autor annotiert, Bearish/Bullish/Neutral | **news_sentiment**, committet |
| [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) | CC BY-NC-SA 3.0 | 2.259 Saetze mit voller Einigkeit von 5 bis 8 Annotatoren | menschlich | zusaetzliches Stimmungs-Testset, **nur Download-Skript** (Share-Alike) |
| [FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) | CC BY-NC 4.0 | 36.537 Schlagzeilen der 20 Paar-Ticker (aus 5,7 GB gestreamt) | keine; Regel-/Teacher-Labels | Pairs-Extra, anonymisiert committet |
| SEC EDGAR Submissions API | gemeinfrei | 4.544 8-K-Filings | Item-Codes der Firmen selbst | Pairs-Extra (echte Schock-Labels) |
| EDT / Trade the Event ([GitHub](https://github.com/Zhihan1996/TradeTheEvent)) | keine Lizenz angegeben, Daten auf Google Drive | 9.721 gelabelte Artikel | menschlich | **nicht verwendet** (Lizenz) |
| FinGPT/fingpt-sentiment-train | keine Lizenz angegeben | 76.772 | Mischung aus anderen Quellen | **nicht verwendet** |

`uv run news-datasets` laedt die beiden MIT-Datensaetze (4 s), mappt sie, bereinigt Texte (HTML-Entities wie `&amp;`, t.co-Links, Leerraum) und dedupliziert ueber eine normalisierte Form.
`uv run news-datasets --phrasebank` baut zusaetzlich das PhraseBank-Testset lokal (`data/out/sets/news_sentiment_phrasebank_test.jsonl`, gitignored).

## Label-Schema und Mapping

Ereignistyp (`tasks/news_event_labels.json`, dort auch die Definitionen):

| Label | Quell-Themen (twitter-financial-news-topic) |
|---|---|
| `earnings` | Earnings |
| `analyst_rating` | Analyst Update |
| `m_and_a` | M&A \| Investments |
| `legal_regulatory` | Legal \| Regulation |
| `management_change` | Personnel Change |
| `dividend` | Dividend |
| `company_news` | Company \| Product News, IPO |
| `stock_movement` | Stock Movement, Stock Commentary |
| `macro_market` | Fed \| Central Banks, Macro, Treasuries \| Corporate Debt, Currencies, Energy \| Oil, Gold \| Metals \| Materials, Markets, Politics |
| `other` | General News \| Opinion |
| (verworfen) | Financials: Sektor-News zu Banken und Versicherern, weder klar Firma noch Makro |

Stimmung (`tasks/news_sentiment_labels.json`): Bullish -> `positive`, Bearish -> `negative`, Neutral -> `neutral`.
`fundamental_shock` (fuer das Pairs-Extra) wird deterministisch abgeleitet: `earnings`, `m_and_a`, `legal_regulatory`, `management_change`.

## Label-Verteilung und feste Sets

Rohdaten nach Mapping und Dedup: news_event 17.873 Zeilen (train 12.673, val 1.408, test 3.792), news_sentiment 11.683 Zeilen (8.389 / 932 / 2.362).
Test kommt aus dem Validation-Split der Quelle und ist fest; val ist ein geseedeter 10-%-Anteil des Quell-Trains.

| news_event | Anteil roh | train | val | test |
|---|---|---|---|---|
| macro_market | 29,5 % | 150 | 30 | 40 |
| company_news | 22,9 % | 150 | 30 | 40 |
| stock_movement | 19,1 % | 150 | 30 | 40 |
| other | 7,9 % | 150 | 30 | 40 |
| earnings | 6,8 % | 150 | 30 | 40 |
| management_change | 3,3 % | 150 | 30 | 40 |
| m_and_a | 3,2 % | 150 | 30 | 40 |
| legal_regulatory | 3,1 % | 150 | 30 | 40 |
| dividend | 2,5 % | 150 | 30 | 40 |
| analyst_rating | 1,8 % | 150 | 25 | 40 |
| **Summe** | | **1.500** | **295** | **400** |

| news_sentiment | Anteil roh | train | val | test |
|---|---|---|---|---|
| neutral | 64,7 % | 450 | 100 | 120 |
| positive | 20,3 % | 450 | 100 | 120 |
| negative | 15,1 % | 450 | 100 | 120 |
| **Summe** | | **1.350** | **300** | **360** |

**Seed-Set fuer den Datenmengen-Vergleich:** `data/out/distil/news_event_small/` hat nur 10 Trainingszeilen pro Label (100 insgesamt, Teilmenge der 1.500), aber dieselbe Testmenge, `config.yaml` und `job_description.json` wie `news_event`.
Damit laesst sich v1 (wenig Daten, distil labs generiert den Rest synthetisch) gegen v2 (alle 1.500 Zeilen) auf demselben Test vergleichen.

Baseline ohne Modell (Keyword-Regeln, `uv run experiment --dry-run`): news_event 30,5 % Accuracy auf dem festen Test, news_sentiment 53,1 %, PhraseBank 65,3 %.
Das ist die Latte, die Teacher und kleines Modell klar schlagen muessen.

## Ablauf vor Ort: Langfuse -> distil labs -> Runpod -> Langfuse

```bash
cd pairs_signal
uv sync
cp .env.example .env   # Keys eintragen, dann: set -a; source .env; set +a

# 0. Daten (liegen fertig im Repo; neu bauen in Sekunden)
uv run news-datasets --phrasebank && uv run export-distil

# 1. Langfuse: Datasets hochladen, Teacher als Referenz
uv run langfuse-datasets --task news_event                       # pairs-news_event-{train,val,test}
uv run experiment --task news_event --model teacher --split test --run-name teacher-test
uv run experiment --task news_sentiment --model teacher --split phrasebank_test   # optional, menschliche Labels

# 2. distil labs (Login: distil auth; distil whoami; distil credits-balance)
distil dataset create --output json --data data/out/distil/news_event | jq -r .id          # -> DS (alle Daten)
# Vergleich wenig Daten: dieselben Schritte mit --data data/out/distil/news_event_small
distil teacher-evaluation create-from-dataset --output json $DS | jq -r .id               # -> TE
distil teacher-evaluation metrics --output json $TE | jq .teacher_performance             # binary = Accuracy
distil dataset generate-synthetic-data-train --smoke --output json $DS | jq -r .id        # Smoke zuerst
distil dataset generate-synthetic-data-train --output json $DS | jq -r .id                # -> DS2
distil slm create-from-dataset --output json $DS2 | jq -r .id                             # -> SLM
distil slm status --output json $SLM | jq -r .status                                      # bis JOB_SUCCESS
distil slm metrics --output json $SLM

# 3. Runpod: Adapter holen, auf HF hochladen, vLLM-Worker mit LoRA
distil slm download --destination model $SLM && (cd model && tar -xf model.tar)
runpodctl hub search vllm                                                                  # -> HUB_ID
uv run runpod-deploy --model-dir model --hf-repo <hf-user>/stock-news-adapter --hub-id $HUB_ID            # druckt die Befehle
uv run runpod-deploy --model-dir model --hf-repo <hf-user>/stock-news-adapter --hub-id $HUB_ID --execute
export RUNPOD_ENDPOINT_ID=<id> RUNPOD_API_KEY=...

# 4. Zurueck in Langfuse: kleines Modell gegen Teacher, Fehler sortieren, v2
uv run experiment --task news_event --model student --split test --client-dir model --run-name slm-v1-test
uv run experiment --task news_event --model student --split val  --client-dir model --run-name slm-v1-val
uv run export-errors slm-v1-val          # Fehler (accuracy == 0) aus Langfuse -> data/out/distil/news_event_v2/
distil dataset create --output json --data data/out/distil/news_event_v2 | jq -r .id       # -> wie Schritt 2 bis 4 -> slm-v2-test
```

Fuer die Stimmung dieselben Schritte mit `news_sentiment`.
Lokal statt Runpod: `vllm serve` wie in `distil slm download` beschrieben und `STUDENT_BASE_URL=http://127.0.0.1:8000/v1`.
Der distil-gehostete Endpoint ist der Fallback (Befehle im Pairs-Extra unten).

## Anwendungsbeispiel / Extra: Pairs Trading

Der urspruengliche Plan, jetzt nur noch ein Anwendungsbeispiel: Ein Pairs-Trade auf einen Spread-Ausreisser wird ausgelassen, wenn die letzte Schlagzeile ein `fundamental_shock` ist.
Die Daten (FNSPID-Schlagzeilen der Paar-Firmen, SEC-8-K-Items, Ausreisser-Faelle, Backtest) bleiben vollstaendig im Repo; Befehle: `uv run prepare --dry-run`, `uv run backtest`.

### Zwei Varianten

**Variante 1: Schlagzeilen-Klassifikation der Paar-Firmen.**
Pro Schlagzeile der Paar-Firmen ein Label aus `earnings_warning`, `guidance_up`, `m_and_a`, `legal_regulatory`, `management_change`, `analyst_rating`, `macro_sector`, `noise`.
`fundamental_shock` wird nicht trainiert, sondern deterministisch abgeleitet: true bei `earnings_warning`, `m_and_a`, `legal_regulatory`, `management_change`.
Definitionen: `tasks/headline_labels.json`.
Labels kommen vom Teacher (Claude Haiku 5.5), viele Beispiele, klare Aufgabe, gut destillierbar.
Strategie: nur handeln, wenn die letzte Schlagzeile kein `fundamental_shock` ist.

**Variante 2: Ausreisser-Faelle.**
Pro Ausreisser Kennzahlen plus Schlagzeile plus SEC-8-K-Items, Label ist der echte Ausgang (`reverts` / `no_revert`), keine Imitation eines grossen Modells.
Ehrlich: Mit drei signifikanten Paaren sind das nur 190 Faelle, fuer Training knapp.
Gut als Benchmark fuer das grosse Modell (`baseline`).

### Ablauf fuer das Pairs-Extra

Die Daten sind fertig im Repo (`data/out/`), Schritt 0 ist nur noetig, um neu zu bauen.

```bash
cd pairs_signal
uv sync
cp .env.example .env   # Keys eintragen, dann: set -a; source .env; set +a

# 0. Daten (nur bei Bedarf neu; ~15 min wegen FNSPID-Stream)
uv run prepare --dry-run          # ohne Key: Schlagzeilen mit Keyword-Regeln gelabelt
uv run label-headlines            # mit ANTHROPIC_API_KEY: Teacher-Labels (~0,5 USD, ~10 min, gecacht)
uv run export-distil              # feste Sets + distil-Ordner neu schreiben
uv run backtest                   # Varianten a/b/b_sec/c auf dem Test-Zeitraum

# 1. Langfuse: Datasets hochladen, Teacher-Baseline als Experiment
uv run langfuse-datasets          # pairs-headlines-{train,val,test,case_test}, pairs-cases-{train,val,test}
uv run experiment --task headlines --model teacher --split test        # Scores accuracy / cost_usd / latency_s je Trace
uv run baseline                   # grosses Modell auf den Ausreisser-Faellen inkl. Trade-Ergebnis

# 2. distil labs (Login: distil auth; distil whoami; distil credits-balance)
distil dataset create --output json --data data/out/distil/headlines | jq -r .id           # -> DS
distil teacher-evaluation create-from-dataset --output json $DS | jq -r .id                # -> TE
distil teacher-evaluation metrics --output json $TE | jq .teacher_performance              # binary = Accuracy
distil dataset generate-synthetic-data-train --smoke --output json $DS | jq -r .id         # Smoke zuerst
distil dataset generate-synthetic-data-train --output json $DS | jq -r .id                 # -> DS2
distil slm create-from-dataset --output json $DS2 | jq -r .id                              # -> SLM
distil slm status --output json $SLM | jq -r .status                                       # bis JOB_SUCCESS
distil slm metrics --output json $SLM
distil slm download-predictions $SLM

# 3. Runpod: Adapter holen, auf HF hochladen, vLLM-Worker mit LoRA
distil slm download --destination model $SLM && (cd model && tar -xf model.tar)
runpodctl hub search vllm                                                                   # -> HUB_ID
uv run runpod-deploy --model-dir model --hf-repo <hf-user>/pairs-signal-adapter --hub-id $HUB_ID          # druckt die Befehle
uv run runpod-deploy --model-dir model --hf-repo <hf-user>/pairs-signal-adapter --hub-id $HUB_ID --execute
export RUNPOD_ENDPOINT_ID=<id> RUNPOD_API_KEY=...
runpodctl serverless health $RUNPOD_ENDPOINT_ID

# 4. Zurueck in Langfuse: kleines Modell gegen Teacher, Fehler sortieren, v2
uv run experiment --task headlines --model student --split test --client-dir model --run-name slm-v1-test
uv run experiment --task headlines --model student --split val  --client-dir model --run-name slm-v1-val
uv run export-errors slm-v1-val           # Fehler aus Langfuse (accuracy == 0) -> data/out/distil/headlines_v2/
distil dataset create --output json --data data/out/distil/headlines_v2 | jq -r .id         # -> DS_V2, dann wie Schritt 2 -> SLM_V2
distil slm download --destination model_v2 $SLM_V2 && (cd model_v2 && tar -xf model.tar)     # wie Schritt 3, eigenes Adapter-Repo
uv run runpod-deploy --model-dir model_v2 --hf-repo <hf-user>/pairs-signal-adapter-v2 --hub-id $HUB_ID --name pairs-slm-v2 --execute
RUNPOD_ENDPOINT_ID=<v2-id> uv run experiment --task headlines --model student --split test --client-dir model_v2 --run-name slm-v2-test
uv run experiment --task headlines --model student --split case_test --client-dir model --run-name slm-v1-cases
uv run backtest --student-run data/out/runs/slm-v1-cases.jsonl
```

Fallback, falls Runpod hakt: distil-gehosteter Endpoint (OpenAI-kompatibel, max. 6 h, stoppt nach 1 h ohne Traffic).

```bash
distil deployment create-from-slm $SLM                                                     # -> DEP, warten auf JOB_SUCCESS
distil inference-endpoint create-from-deployment --name pairs-slm --fallback-model "openai/gpt-4.1-mini" $DEP
distil api-keys create pairs-key && distil inference-endpoint link-api-key <endpoint-name> pairs-key
export STUDENT_BASE_URL=https://inference.distillabs.ai/v1 STUDENT_MODEL=<endpoint-name> STUDENT_API_KEY=<key>
```

Hinweise zum Ablauf:

- Das Test-Set (`data/out/sets/*_test.jsonl`, in Langfuse `pairs-*-test`) ist fest und wird nie trainiert; `export-errors` verweigert test-Laeufe.
Die Verbesserungsschleife nutzt den Validierungs-Zeitraum.
- distil labs kennt keinen `classification`-Task mehr; Labels laufen als `question-answering` mit einem Mutator ueber die Labels, Metrik `binary` = Accuracy.
- Der Student braucht den Trainings-System-Prompt (= `task_description` aus `job_description.json`); `--client-dir model` nutzt den `model_client.py` von distil labs mit den richtigen Sampling- und Thinking-Einstellungen.
- Kosten pro Entscheidung: Teacher aus Token-Usage und Listenpreis, Student aus Latenz mal `RUNPOD_PRICE_PER_SEC` (Annahme, an den gebuchten GPU-Tarif anpassen).
- `runpodctl --env` parst Werte als CSV; `runpod-deploy` quotet `LORA_MODULES` deshalb korrekt (unquotiertes JSON bricht ab).
- `--workers-min 1` haelt einen Worker warm und kostet sofort; nach der Demo `runpodctl serverless delete <id>`.

### Datensatz

Stand 2026-10-19, `uv run prepare --dry-run` (Laufzeit ohne Downloads 11 s; FNSPID-Stream 15 min, SEC 1 min, Kurse 10 s).

**Quellen:** 3522 Handelstage x 20 Ticker (2010 bis 2023), 4544 SEC-8-K-Filings, 36.537 FNSPID-Schlagzeilen.
FNSPID `All_external.csv` endet fuer diese Ticker am 2020-06-11 und hat keine Schlagzeilen zu GS und CVX (UPS, V, WMT erst ab 2020).
Die Splits liegen deshalb innerhalb der News-Abdeckung: train bis 2017-12-31, val 2018, test 2019-01-01 bis 2020-06-11 (inklusive Corona-Crash).
`fetch-news --source nasdaq` streamt die 23-GB-Datei bis 2023 (ca. 1 h), danach koennen die Grenzen in `config.py` nach hinten.

**Paare** (`data/out/pairs_summary.csv`, Engle-Granger ueber 2010 bis 2023):

| Paar | p-Wert | Hedge-Ratio | Halbwertszeit (Median, Tage) | Status |
|---|---|---|---|---|
| GS/MS | 0,001 | 0,69 | 18,7 | Kern |
| CAT/DE | 0,028 | 0,71 | 20,9 | Kern |
| KO/PEP | 0,034 | 0,74 | 18,8 | Kern |
| HD/LOW | 0,109 | 1,04 | 22,2 | nicht signifikant, nur zur Illustration |
| V/MA | 0,135 | 0,92 | 11,9 | nicht signifikant, nur zur Illustration |
| JPM/BAC, WMT/TGT, UPS/FDX, XOM/CVX, PG/CL | 0,21 bis 0,79 | | | ausgeschlossen |

**Schlagzeilen (Hauptweg):** 29.147 eindeutige, anonymisierte Schlagzeilen (train 15.964, val 4.018, test 9.165).
Verteilung mit Keyword-Regeln (Platzhalter bis zum Teacher-Lauf): noise 71,5 %, analyst_rating 12,0 %, macro_sector 7,2 %, guidance_up 3,2 %, earnings_warning 2,2 %, m_and_a 2,0 %, legal_regulatory 1,6 %, management_change 0,3 %; `fundamental_shock` 6,1 %.
Feste, pro Label balancierte Sets fuer distil labs und Langfuse: train 1.092 (max. 150 je Label), val 301 (max. 40), test 375 (max. 50), dazu 2.000 weitere train-Schlagzeilen als `traces.jsonl`.
`management_change` ist selten (42 / 21 / 25), dort ist der Mutator von distil labs wichtig.
Teacher-Labeling aller Schlagzeilen kostet geschaetzt 0,3 bis 0,5 USD (Haiku 5.5, 25 Schlagzeilen pro Aufruf).

**Ausreisser-Faelle (Kernpaare):** 190 Faelle, 95 % mit Schlagzeile.

| Split | Faelle | reverts | no_revert | Basisquote reverts |
|---|---|---|---|---|
| train | 143 | 22 | 121 | 15 % |
| val | 16 | 4 | 12 | 25 % |
| test | 31 | 7 | 24 | 23 % |

**Backtest Test-Zeitraum** (`uv run backtest`, Schlagzeilen-Labels hier noch aus Regeln):

| Variante | Trades | Trefferquote | Rendite pro Trade | Summe |
|---|---|---|---|---|
| a naiv, jeder Ausreisser | 31 | 22,6 % | +0,55 % | +17,2 % |
| b ohne fundamental_shock-Schlagzeile (Regeln) | 30 | 20,0 % | +0,40 % | +12,1 % |
| b_sec ohne fundamentales 8-K (echte SEC-Daten) | 22 | 31,8 % | +0,80 % | +17,7 % |
| c kleines Modell | Platzhalter, nach Runpod-Deploy | | | |

Die SEC-Variante ist der modellfreie Beleg fuer die These: Ausreisser mit fundamentalem 8-K in den 7 Tagen davor kommen seltener zurueck (ueber alle 190 Kernfaelle 9,2 % von 65 gegen 21,6 % von 125), ebenso bei Volumen-Spike > 2x (13,5 % gegen 18,8 %).
Passend dazu der Befund von Deutsche Bank / RavenPack (FactSet Insight, 2016): ein News-Filter senkte nicht konvergierende Paare von 15 % auf 7 %.
Mit 31 Test-Faellen sind diese Zahlen Anschauung, kein Beweis.

### Methode

- **Kurse:** yfinance, adjustierte Schlusskurse und Volumen 2010 bis 2023, 10 klassische Paare; Faelle nur bis zum Ende der News-Abdeckung.
- **Paare:** Engle-Granger (statsmodels) auf Log-Kursen.
Kern-Backtest nur mit den Paaren mit p < 0,05 ueber den ganzen Zeitraum.
HD/LOW und V/MA (p 0,11 bis 0,13) sind **nicht signifikant, nur zur Illustration** (`build-cases --pairs illustrative`).
- **Spread und z-Score ohne Lookahead:** Hedge-Ratio, Achsenabschnitt und Kointegrationstest alle 21 Tage auf den letzten 252 Tagen neu geschaetzt und fuer den naechsten Block eingefroren; z-Score mit rollierendem 60-Tage-Mittel und -Std.
- **Ereignisse:** ein Fall = erster Tag mit |z| >= 2; ein neuer Fall erst, wenn |z| vorher unter 0,5 war oder 10 Handelstage vergangen sind.
So ist ein Ausreisser einmal im Datensatz und nicht zehnmal.
- **Label:** Mit am Ereignistag eingefrorenen Parametern: `reverts`, wenn |z| innerhalb von 10 Handelstagen unter 0,5 faellt, sonst `no_revert`.
- **Trade:** Spread am Ereignis-Schluss gegen die Richtung von z eroeffnen, Ausstieg bei |z| < 0,5 oder nach 10 Tagen, Rendite auf Brutto-Nominal (1 + |Hedge-Ratio|), Kosten 2 x 5 bp.
- **Kennzahlen pro Fall:** z, z am Vortag, Halbwertszeit (AR(1) auf dem Spread), Korrelation der Tagesrenditen, rollierender Engle-Granger-p-Wert, Dauer der Kointegration, Hedge-Ratio, Tagesrenditen, Volumen-Spike je Firma (Volumen / Median der 20 Vortage), SEC-8-K-Items der letzten 7 Tage, letzte Schlagzeile.
- **Schlagzeilen:** FNSPID (Hugging Face, Nasdaq-News mit Datum und Ticker, CC BY-NC 4.0).
Die 5,7-GB-Datei `Stock_news/All_external.csv` wird gestreamt und nur auf die 20 Ticker gefiltert (wenige MB bleiben), mit Resume per HTTP-Range und Byte-Pruefung.
Nur Schlagzeilen, die vor dem Ereignis-Schluss (20:00 UTC) bekannt waren, hoechstens 7 Tage alt.
- **SEC 8-K:** EDGAR Submissions API (`data.sec.gov`), Item-Codes mit `acceptanceDateTime` gegen Lookahead.
Fundamental-Items (u. a. 2.02 Ergebnisse, 5.02 Management, 1.01/2.01 Vertraege und M&A) liefern echte, modellfreie Schock-Labels fuer Backtest-Variante `b_sec`.
- **Anonymisierung:** Firmennamen, Ticker (auch in Ticker-Listen) und Boersen-Tags werden zu "Company A" / "Company B", Jahreszahlen zu `[year]`, Monatsdaten zu `[date]`; Fall-Texte enthalten keine Daten, nur relative Tage.
Dateien mit echten Tickern und Daten bleiben in `data/interim/` (gitignored).
- **Splits nach Zeit:** train bis 2017-12-31, val 2018, test 2019-01-01 bis 2020-06-11 (juengster Zeitraum mit Schlagzeilen).
Faelle, deren 10-Tage-Fenster in den naechsten Split reicht, werden verworfen (Embargo); Schlagzeilen sind ueber den Text dedupliziert, keine liegt in zwei Splits.

### Ehrliche Grenzen des Backtests

- Wenige Faelle: drei Paare, ein Test-Zeitraum von 1,5 Jahren mit 31 Faellen, Trefferquoten auf so wenigen Trades schwanken stark.
Der Test-Zeitraum enthaelt den Corona-Crash 2020.
- Kointegration ist instabil; die Paarauswahl nutzt den ganzen Zeitraum (inklusive Test) und ist damit selbst ein Lookahead.
Ehrlicher waere eine Auswahl nur auf dem Trainings-Zeitraum.
- Rendite ist eine Naeherung: Log-Spread-Aenderung auf Brutto-Nominal, keine Leerverkaufskosten, kein Slippage ueber die pauschalen 5 bp hinaus, Einstieg zum Schlusskurs des Signaltags.
- Die Label-Definition (zurueck unter |z| < 0,5 in 10 Tagen) ist streng; bei Halbwertszeiten um 20 Tage ist die Basisquote niedrig.
- FNSPID-Abdeckung schwankt nach Jahr und Ticker; nicht jeder Fall hat eine Schlagzeile.
- Ohne API-Key stammen die Schlagzeilen-Labels aus Keyword-Regeln (`label_source: rules`); erst `label-headlines` mit Key liefert Teacher-Labels.
Die 50 als `review: true` markierten Faelle (`data/out/headline_review.csv`) sind zur manuellen Kontrolle gedacht.

## Daten und Lizenzen

Das Repo ist oeffentlich; committet sind nur Daten, deren Lizenz das erlaubt.

- **Ereignistyp und Stimmung:** [zeroshot/twitter-financial-news-topic](https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic) und [zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment), Lizenz MIT; die gemappten Zeilen in `data/out/news/` und die Sets sind daraus abgeleitet.
- **Financial PhraseBank:** Malo et al. 2014, [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank), [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/); nicht committet, nur per `news-datasets --phrasebank` lokal gebaut.
- **Pairs-Extra, Schlagzeilen:** abgeleitet aus FNSPID (Dong, Fan, Peng 2024, [arXiv 2402.06698](https://arxiv.org/abs/2402.06698)), Hugging Face [Zihan1004/FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID), Lizenz [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Die Schlagzeilen in `data/out/` sind anonymisiert und gefiltert, bleiben aber abgeleitete Daten unter CC BY-NC 4.0: nur nicht-kommerzielle Nutzung, mit Namensnennung.
- **SEC EDGAR:** 8-K-Filings ueber die Submissions API (`data.sec.gov`), Daten einer US-Bundesbehoerde und gemeinfrei.
- **Kurse und Volumen:** via yfinance von Yahoo Finance, es gelten die Nutzungsbedingungen von Yahoo; nur fuer Forschung und Lehre.
Rohkurse liegen nur lokal in `data/raw/` (gitignored); im Repo stehen nur daraus berechnete Kennzahlen.

## Dateien

| Pfad | Inhalt |
|---|---|
| `src/pairs_signal/` | Pipeline-Module, je ein CLI-Eintrag in `pyproject.toml` |
| `tasks/news_event_labels.json`, `tasks/news_sentiment_labels.json` | Produkt: Aufgabenbeschreibung, Label-Definitionen, Quell-Mapping |
| `data/out/news/{news_event,news_sentiment}.jsonl` | alle gemappten, deduplizierten Zeilen mit Split und Quell-Label |
| `data/out/distil/{news_event,news_sentiment}/` | Produkt: Upload-Ordner fuer `distil dataset create --data` |
| `data/out/distil/news_event_small/` | Seed-Set mit 10 Trainingszeilen pro Label, gleiche Testmenge und Config |
| `tasks/headline_labels.json` | Aufgabenbeschreibung, Label-Definitionen, `fundamental_shock`-Regel |
| `data/out/headlines.jsonl` | alle anonymisierten Schlagzeilen mit Split und Label |
| `data/out/cases.jsonl` | anonymisierte Ausreisser-Faelle mit Kennzahlen, Label und Trade-Ergebnis |
| `data/out/sets/` | feste train/val/test-Sets (`{id, input, label, meta}`), Quelle fuer Langfuse und Experimente |
| `data/out/distil/{headlines,cases}/` | Pairs-Extra: Upload-Ordner fuer `distil dataset create --data` |
| `data/out/headline_review.csv` | 50 Stichproben zur manuellen Label-Kontrolle |
| `data/out/backtest_test.json` | Backtest-Ergebnis des letzten Laufs |
| `data/out/pairs_summary.csv` | Paar-Tabelle (p-Wert, Hedge-Ratio, Status) |

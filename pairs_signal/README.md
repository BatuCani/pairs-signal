# pairs-signal

Hackathon 2026-10-19 (distil labs, Langfuse, Runpod): ein kleines Modell bewertet Pairs-Trading-Ausreisser anhand der Nachrichtenlage.
Python rechnet Paare, Spreads und Ausreisser vorher aus.
Das kleine Modell klassifiziert die letzte Schlagzeile vor dem Ausreisser, und eine feste Regel entscheidet daraus, ob gehandelt wird.

Dieses Projekt liegt im Unterordner `pairs_signal/` des Repos und ist ein eigenes uv-Projekt; alle Befehle und Pfade unten gelten in diesem Ordner (`cd pairs_signal`).
Der Root des Repos gehoert zum separaten Projekt "Pair Research Agent".

Alles laeuft ohne Keys mit `--dry-run`.
Keys kommen nur aus der Umgebung (`.env.example` zeigt alle Variablen).

## Zwei Wege

**Hauptweg (empfohlen): Schlagzeilen-Klassifikation.**
Pro Schlagzeile der Paar-Firmen ein Label aus `earnings_warning`, `guidance_up`, `m_and_a`, `legal_regulatory`, `management_change`, `analyst_rating`, `macro_sector`, `noise`.
`fundamental_shock` wird nicht trainiert, sondern deterministisch abgeleitet: true bei `earnings_warning`, `m_and_a`, `legal_regulatory`, `management_change`.
Definitionen: `tasks/headline_labels.json`.
Labels kommen vom Teacher (Claude Haiku 5.5), viele Beispiele, klare Aufgabe, gut destillierbar.
Strategie: nur handeln, wenn die letzte Schlagzeile kein `fundamental_shock` ist.

**Nebenweg: Ausreisser-Faelle.**
Pro Ausreisser Kennzahlen plus Schlagzeile plus SEC-8-K-Items, Label ist der echte Ausgang (`reverts` / `no_revert`), keine Imitation eines grossen Modells.
Ehrlich: Mit drei signifikanten Paaren sind das nur 190 Faelle, fuer Training knapp.
Gut als Benchmark fuer das grosse Modell (`baseline`).

## Ablauf vor Ort: Langfuse -> distil labs -> Runpod -> Langfuse

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
uv run experiment --model teacher --split test        # Scores accuracy / cost_usd / latency_s je Trace
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
uv run experiment --model student --split test --client-dir model --run-name slm-v1-test
uv run experiment --model student --split val  --client-dir model --run-name slm-v1-val
uv run export-errors slm-v1-val           # Fehler aus Langfuse (accuracy == 0) -> data/out/distil/headlines_v2/
distil dataset create --output json --data data/out/distil/headlines_v2 | jq -r .id         # -> DS_V2, dann wie Schritt 2 -> SLM_V2
distil slm download --destination model_v2 $SLM_V2 && (cd model_v2 && tar -xf model.tar)     # wie Schritt 3, eigenes Adapter-Repo
uv run runpod-deploy --model-dir model_v2 --hf-repo <hf-user>/pairs-signal-adapter-v2 --hub-id $HUB_ID --name pairs-slm-v2 --execute
RUNPOD_ENDPOINT_ID=<v2-id> uv run experiment --model student --split test --client-dir model_v2 --run-name slm-v2-test
uv run experiment --model student --split case_test --client-dir model --run-name slm-v1-cases
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

## Datensatz

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

## Methode

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

## Ehrliche Grenzen des Backtests

- Wenige Faelle: drei Paare, ein Test-Zeitraum von 1,5 Jahren mit 31 Faellen, Trefferquoten auf so wenigen Trades schwanken stark.
Der Test-Zeitraum enthaelt den Corona-Crash 2020.
- Kointegration ist instabil; die Paarauswahl nutzt den ganzen Zeitraum (inklusive Test) und ist damit selbst ein Lookahead.
Ehrlicher waere eine Auswahl nur auf dem Trainings-Zeitraum.
- Rendite ist eine Naeherung: Log-Spread-Aenderung auf Brutto-Nominal, keine Leerverkaufskosten, kein Slippage ueber die pauschalen 5 bp hinaus, Einstieg zum Schlusskurs des Signaltags.
- Die Label-Definition (zurueck unter |z| < 0,5 in 10 Tagen) ist streng; bei Halbwertszeiten um 20 Tage ist die Basisquote niedrig.
- FNSPID-Abdeckung schwankt nach Jahr und Ticker; nicht jeder Fall hat eine Schlagzeile.
- Ohne API-Key stammen die Schlagzeilen-Labels aus Keyword-Regeln (`label_source: rules`); erst `label-headlines` mit Key liefert Teacher-Labels.
Die 50 als `review: true` markierten Faelle (`data/out/headline_review.csv`) sind zur manuellen Kontrolle gedacht.

## Dateien

| Pfad | Inhalt |
|---|---|
| `src/pairs_signal/` | Pipeline-Module, je ein CLI-Eintrag in `pyproject.toml` |
| `tasks/headline_labels.json` | Aufgabenbeschreibung, Label-Definitionen, `fundamental_shock`-Regel |
| `data/out/headlines.jsonl` | alle anonymisierten Schlagzeilen mit Split und Label |
| `data/out/cases.jsonl` | anonymisierte Ausreisser-Faelle mit Kennzahlen, Label und Trade-Ergebnis |
| `data/out/sets/` | feste train/val/test-Sets (`{id, input, label, meta}`), Quelle fuer Langfuse und Experimente |
| `data/out/distil/{headlines,cases}/` | Upload-Ordner fuer `distil dataset create --data` |
| `data/out/headline_review.csv` | 50 Stichproben zur manuellen Label-Kontrolle |
| `data/out/backtest_test.json` | Backtest-Ergebnis des letzten Laufs |
| `data/out/pairs_summary.csv` | Paar-Tabelle (p-Wert, Hedge-Ratio, Status) |

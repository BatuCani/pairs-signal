# Start in PyCharm

1. Öffne diesen Ordner als Projekt in PyCharm.
2. Wähle als Python-Interpreter `.venv\Scripts\python.exe` im Projektordner.
3. Öffne das PyCharm-Terminal und starte die Offline-Demo:

```powershell
.\.venv\Scripts\python.exe -m pair_agent.cli research --ticker KO --synthetic
```

Für echte Yahoo-Marktdaten:

```powershell
.\.venv\Scripts\python.exe -m pair_agent.cli research --ticker NVDA
```

Alle 20 Evaluation-Aufgaben offline ausführen:

```powershell
.\.venv\Scripts\python.exe scripts/run_evaluation.py --synthetic
```

Tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Es werden keine Trades ausgeführt. Die synthetischen Daten sind erfunden.
Ein Ergebnis ohne robustes Paar ist ausdrücklich vorgesehen. Zugangsdaten werden
nicht benötigt; externe Integrationen sind in der README als nächste Schritte beschrieben.

# News Gate - pitch script (60 seconds)

Slides: `pitch/slides.html`, 3 slides (arrow keys, **N** speaker notes, **F** full screen).
Numbers in [brackets] come from the `numbers` block at the top of `slides.html`; if one is still TBD, drop that clause.
About 150 spoken words; Batu speaks, the teammate clicks.

## 0:00 - Problem and product (slide 1)

Trading agents read thousands of stock headlines a day, and they need to know in real time: real news, or noise?
Keyword rules get only [rules accuracy] of ten event types right.
A frontier model for every headline is slow and expensive.
So we built News Gate: our own tiny model that labels every headline, like this one: Coca-Cola gets a new CFO, management change, a real shock.

## 0:20 - The loop (slide 2)

We collect traces on Langfuse, train the model on distil labs, and run it on Runpod.
Langfuse scores every answer against human labels, and the misses train version two: from [v1] to [v2] accuracy, on a test set the model never saw.

## 0:40 - Teacher vs. tiny model (slide 3)

The result: frontier quality at a fraction of the cost.
[tiny accuracy] versus [teacher accuracy] for the teacher, at [tiny cost] instead of [teacher cost] per thousand calls, in [tiny latency], on our own GPU.
News Gate.
Thank you.

## Kurz auf Deutsch

- **0:00 Folie 1:** Agenten lesen tausende Schlagzeilen; Keyword-Regeln treffen nur [rules accuracy] (aktuell 30,5 %), Frontier-Modell pro Schlagzeile zu langsam und teuer; News Gate = eigenes kleines Modell, Beispiel Coca-Cola-CFO = management_change.
- **0:20 Folie 2:** Sponsor-Satz woertlich: Traces auf Langfuse, Training auf distil labs, laufen auf Runpod; Fehler von v1 trainieren v2, Testset fest.
- **0:40 Folie 3:** "Frontier quality at a fraction of the cost": Genauigkeit, Kosten pro 1.000 Aufrufe, Latenz, eigene GPU. Danke.
- **Ehrlich:** nur Zahlen sagen, die im JSON-Block stehen; Pairs-Backtest nur auf Nachfrage (siehe `jury-qa.md`).

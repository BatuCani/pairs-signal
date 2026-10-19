# News Gate - pitch script (60 seconds)

Slides: `pitch/slides.html`, 3 slides (arrow keys, **N** speaker notes, **F** full screen).
All numbers are from the distil labs evaluation on the 400-headline test set (exact category match); they match the `numbers` block in `slides.html`.
About 150 spoken words; Batu speaks, the teammate clicks.

## 0:00 - Problem and product (slide 1)

Trading agents read thousands of stock headlines a day, and they need to know in real time: real news, or noise?
Keyword rules get only 30.5 percent of ten event types right.
A frontier model for every headline is slow and expensive.
So we built News Gate: our own tiny model that labels every headline, like this one: Coca-Cola gets a new CFO, management change, a real shock.

## 0:20 - The loop (slide 2)

We collect traces on Langfuse, train the model on distil labs, and run it on Runpod.
Langfuse scores every answer.
Out of the box, Qwen 3.5 with 0.8 billion parameters gets 55.75 percent; after six and a half minutes of training on distil labs, version one gets 88 percent.
Version two adds 1,000 synthetic headlines from distil labs and is training right now.

## 0:40 - Teacher vs. tiny model (slide 3)

The result: frontier quality at a fraction of the cost.
Our 0.8 billion parameter student beats its own teacher: 88 percent versus 79 percent for GLM 5.3, on 400 test headlines it never saw.
A tiny model, on our own GPU.
News Gate.
Thank you.

## Kurz auf Deutsch

- **0:00 Folie 1:** Agenten lesen tausende Schlagzeilen; Keyword-Regeln treffen nur 30,5 %, Frontier-Modell pro Schlagzeile zu langsam und teuer; News Gate = eigenes kleines Modell, Beispiel Coca-Cola-CFO = management_change.
- **0:20 Folie 2:** Sponsor-Satz woertlich: Traces auf Langfuse, Training auf distil labs, laufen auf Runpod; untrainiert 55,75 %, Version 1 nach 6,6 min Training 88,0 %; v2 mit 1.000 synthetischen Schlagzeilen trainiert noch.
- **0:40 Folie 3:** 0,8B-Schueler schlaegt den Lehrer: 88 % gegen 79,25 % (GLM 5.3), 400 Testschlagzeilen. "Frontier quality at a fraction of the cost". Danke.
- **Ehrlich:** nur Zahlen sagen, die im JSON-Block stehen; Pairs-Backtest nur auf Nachfrage (siehe `jury-qa.md`).

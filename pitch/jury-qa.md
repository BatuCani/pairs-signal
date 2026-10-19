# News Gate - likely jury questions

Short, honest answers, each under 20 seconds.
Numbers in [brackets] come from the `numbers` block in `slides.html`; only say them once they are filled in.
Sources for every fixed number: `pairs_signal/README.md` on branch `fm/pairs-prep`.

## 1. Why AI at all? Wouldn't keyword rules do?

We tried.
Our keyword baseline gets 30.5 percent of the ten event types right on the fixed test set of 400 headlines, and calls a new Coca-Cola CFO "other".
Headlines are short and full of tickers and jargon; you need language understanding, not word lists.
In the pairs backtest, keyword rules even made the strategy worse than doing nothing (20 versus 23 percent hit rate).

## 2. Why a tiny model instead of just calling a big one?

Three reasons: cost, speed and privacy.
A market feed is thousands of headlines a day per watchlist, so cost per call decides whether this runs at all: [tiny cost] versus [big cost] per thousand calls.
Latency is [tiny latency] versus [big latency].
And it runs on our own GPU, so which stocks we watch never goes to a third-party API.
The trade-off is accuracy: [tiny accuracy] versus [big accuracy] on the same test set.

## 3. How do you know the labels are right? Isn't the model just copying the teacher?

The product labels are not from an LLM.
They come from two open datasets labeled by people (twitter-financial-news topic and sentiment, MIT license), mapped onto our ten event types and three sentiments.
Both the tiny and the big model are scored against those human labels on a fixed, label-balanced test set the model never trains on.
For sentiment there is an extra human-labeled check, the Financial PhraseBank (only downloaded locally because of its share-alike license).

## 4. What is the base rate? Is your accuracy actually good?

Ten classes, balanced test set: random guessing is 10 percent, our keyword baseline 30.5 percent.
For sentiment: three classes, keyword baseline 53.1 percent, PhraseBank 65.3 percent.
That is the bar both models have to clear, and we name it in the pitch.
For the pairs example the base rate is 23 percent: trading every outlier in the test period reverted 23 percent of the time.

## 5. Any lookahead or leakage?

For the classifier: the test set comes from the dataset's own validation split, is fixed, and is never used for training; texts are deduplicated on a normalized form, and test wins, so the same headline cannot sit in train and test.
The improvement loop only feeds back errors from the validation split; `export-errors` refuses test runs.
For the pairs backtest: hedge ratio and spread are re-estimated every 21 days on past data only, and headlines must be published before the close of the outlier day.
One honest leak remains there: the choice of the three pairs used the whole period, including the test years.

## 6. Does it generalize beyond these datasets?

Honestly, not proven yet.
The training data is English financial news tweets and headlines; different sources (long articles, other languages, other markets) would need their own test set.
The loop is the answer: every live call lands in Langfuse, misses become training examples, distil labs trains the next version.
What generalizes is the recipe, not this exact checkpoint.

## 7. What does it cost to run, really?

Tiny model: latency times the Runpod price per GPU-second, [tiny cost] per thousand calls in our measurements (the price per second is set to the booked GPU tier).
Big model: tokens times list price, [big cost] per thousand calls.
Training with distil labs and a serverless Runpod worker means no GPU is paid while nothing runs; a warm worker costs while it idles.

## 8. How is this different from RavenPack or Bloomberg news analytics?

Those are excellent, closed data products you subscribe to.
News Gate is your own model: you choose the labels, you see every decision in Langfuse, you retrain it on your own mistakes, and it runs on your own GPU.
It fits inside an agent as a cheap tool call rather than a data feed.

## 9. The pairs backtest is only 31 trades. Isn't that meaningless?

It is an illustration, not proof, and it is not part of the product claim.
That is why the main evidence is the classifier accuracy on 400 human-labeled test headlines.
The pairs part shows why the labels matter: across all 190 outliers, the ones with a fundamental SEC 8-K filing came back only 9 percent of the time (65 cases), without one 22 percent (125 cases).
The external evidence is bigger: Deutsche Bank and RavenPack cut non-converging pairs from 15 to 7 percent with a news filter (FactSet Insight, 2016).

## 10. What did you actually build today, and what was there before?

[Fill in before the pitch, honestly: what existed before the event started.]
The pieces of the product: the label mapping and fixed test sets on open datasets, the distil labs training runs, the Runpod deployment, Langfuse tracing and scoring, the v1 to v2 loop, and the demo script.
The teammate built the Pair Research Agent that calls News Gate as a tool.
No live trading, no orders, not investment advice.

import re

from pair_agent.data.universe import Stock
from pair_agent.models.schemas import ResearchRequest


def find_peer_candidates(ticker: str, universe: list[Stock]) -> list[str]:
    target = next((s for s in universe if s.ticker == ticker), None)
    if not target:
        return []
    peers = [s for s in universe if s.ticker != ticker and s.sector == target.sector]
    peers.sort(key=lambda s: (s.industry != target.industry, s.ticker))
    return [s.ticker for s in peers]


def resolve_request(request: ResearchRequest, universe: list[Stock]) -> ResearchRequest:
    if request.ticker or request.tickers:
        return request
    text = request.request
    mentioned = [s.ticker for s in universe if re.search(rf"(?<![A-Z]){re.escape(s.ticker)}(?![A-Z])", text)]
    # Ticker mentions must be uppercase, avoiding accidental matches on words like 'a'.
    count = re.search(r"\b([1-9]|1[0-9]|20)\b", text)
    words = {"three": 3, "five": 5, "drei": 3, "fünf": 5, "two": 2}
    top_k = int(count.group()) if count else next((v for w, v in words.items() if w in text.lower()), request.top_k)
    if len(mentioned) == 1:
        return request.model_copy(update=dict(ticker=mentioned[0], top_k=top_k))
    if not mentioned:
        selectors = {"bank": "Banks", "semiconductor": "Semiconductors", "energy": "Energy",
                     "beverage": "Beverages", "healthcare": "Healthcare", "software": "Software",
                     "staples": "Consumer Staples"}
        label = next((label for word, label in selectors.items() if word in text.lower()), None)
        mentioned = [s.ticker for s in universe if not label or s.sector == label or s.industry == label]
    return request.model_copy(update=dict(tickers=mentioned, top_k=top_k))


def candidate_pairs(request: ResearchRequest, universe: list[Stock]) -> list[tuple[str, str]]:
    lookup = {s.ticker: s for s in universe}
    if request.ticker:
        allowed = set(request.tickers) if request.tickers else set(lookup)
        return [(request.ticker, p) for p in find_peer_candidates(request.ticker, universe) if p in allowed]
    tickers = request.tickers or list(lookup)
    pairs = []
    for i, a in enumerate(tickers):
        for b in tickers[i+1:]:
            if a in lookup and b in lookup and lookup[a].sector == lookup[b].sector:
                pairs.append((a, b))
    return sorted(pairs, key=lambda p: (lookup[p[0]].industry != lookup[p[1]].industry, p))

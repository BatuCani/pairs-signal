import pandas as pd

from pair_agent.config import ResearchConfig


def chronological_split(prices: pd.DataFrame, config: ResearchConfig) -> dict[str, pd.DataFrame]:
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("prices must have a unique ascending date index")
    train_end, validation_end = config.boundaries()
    prices = prices.loc[(prices.index >= pd.Timestamp(config.start)) &
                        (prices.index < pd.Timestamp(config.end))]
    return {"train": prices.loc[prices.index <= train_end],
            "validation": prices.loc[(prices.index > train_end) & (prices.index <= validation_end)],
            "test": prices.loc[prices.index > validation_end]}

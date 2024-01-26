from src.agg_strategy import (
    Strategy,
    FedAvgStrategy,
    LocalStrategy
)


def load_agg_strategy(name: str, agg_strategy_params) -> Strategy:

    if name == 'fedavg':
        return FedAvgStrategy(**agg_strategy_params)
    elif name == 'local':
        return LocalStrategy(**agg_strategy_params)
    else:
        raise ValueError(f'agg_strategy {name} is not supported')

def parse_strategy_params(strategy):
    """
    Parse strategy and params from string.
    example: 'strategy@key1=value1-key2=value2' -> ('strategy', {'key1': 'value1', 'key2': 'value2'})
    """
    if '@' in strategy:
        strategy, params = strategy.split('@')
        param_dict = {}
        if params == '':
            return strategy, {}
        elif '-' in params:
            ret = params.split('-')
            for item in ret:
                if '=' in item:
                    key, value = item.split('=')
                    param_dict[key] = value
        elif '=' in params:
            key, value = params.split('=')
            param_dict[key] = value
        else:
            raise ValueError(f"invalid format of params, example: strategy@key1=value1-key2=value2")
        return strategy, param_dict
    else:
        return strategy, {}

from src.utils.tracker import Tracker


class ResultAnalyzer:
    def __init__(self, params: dict):
        pass

    def clean_and_analyze_results(self, tracker: Tracker) -> dict:
        return tracker.to_dict()

    # return {
    #     'rets': rets,
    #     'history': history,
    #     'eval_history': eval_history,
    # }
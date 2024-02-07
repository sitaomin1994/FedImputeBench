
class Tracker:

    def __init__(self):

        self.imp_quality = []  # tracking history results of imputation quality
        self.data = []  # tracking final imputed data
        self.model_params = []  # TODO: model parameters tracking (intervals)

    def to_dict(self):
        return {
            "results":{
                'imp_quality': self.imp_quality,
            },
            'data': {
                "imp_data": self.data,
                "model_params": self.model_params
            }
        }
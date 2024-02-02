
class Tracker:

    def __init__(self):

        self.imp_quality = []  # tracking history results of imputation quality
        self.data = []  # tracking final imputed data
        self.model_params = []

    def to_dict(self):
        return {
            'imp_quality': self.imp_quality,
            'data': self.data
        }
from abc import ABC, abstractmethod

import numpy as np
from src.imputation.base import BaseImputer


class JMImputer(BaseImputer):

    def __init__(self):
        super().__init__()

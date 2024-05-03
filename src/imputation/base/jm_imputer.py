from abc import ABC, abstractmethod

import numpy as np
from src.imputation.base import BaseImputer


class JMImputerMixin:

    def __init__(self):
        super().__init__()

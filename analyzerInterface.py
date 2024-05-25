from abc import ABC, abstractmethod


class AnalyzerInterface(ABC):
    def __init__(self):
        self.df = None
        self.rendered_result = None

    @abstractmethod
    def setup_data(self):
        pass

    @abstractmethod
    def analyze(self):
        pass

    @abstractmethod
    def render_result(self):
        pass

    @abstractmethod
    def get_rendered_result(self):
        pass

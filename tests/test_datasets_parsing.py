import unittest

from pygraphs.graphs import Datasets


class TestDatasetsParsing(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datasets = Datasets()

    def test_load_polbooks(self):
        _ = self.datasets['polbooks']
        _ = self.datasets.polbooks

    def test_load_zachary(self):
        _ = self.datasets['zachary']
        _ = self.datasets.zachary

    def test_load_newsgroup(self):
        _ = self.datasets['news_2cl_1']
        _ = self.datasets.news_2cl_1

    def test_load_webkb(self):
        _ = self.datasets['webkb_cornel']
        _ = self.datasets.webkb_cornel

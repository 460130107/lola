"""
:Authors: - Wilker Aziz
"""

import unittest

from lola.sparse import SparseCategorical


class SparseCategoricalTestCase(unittest.TestCase):


    def test_zero(self):
        c1 = SparseCategorical(100)
        self.assertEqual(100, c1.support_size())
        self.assertEqual(0, c1.n_represented())
        self.assertEqual(0, c1.sum())
    
    def test_one(self):
        c1 = SparseCategorical(100, 1.0)
        self.assertEqual(100, c1.support_size())
        self.assertEqual(0, c1.n_represented())
        self.assertEqual(100, c1.sum())
    
    def test_mixed(self):
        c1 = SparseCategorical(100, 1.0)
        self.assertEqual(100, c1.support_size())
        self.assertEqual(0, c1.n_represented())
        c1.acc(5, 2.0)
        self.assertEqual(102, c1.sum())
    
    def test_normalise(self):
        c1 = SparseCategorical(10)
        c1.acc(0, 5.0)
        c1.acc(0, 5.0)
        c1.acc(1, 2.0)
        c1.acc(9, 8.0)
        self.assertEqual(20.0, c1.sum())
        c1.normalise()
        self.assertEqual(1.0, c1.sum())
        print(c1)


if __name__ == '__main__':
    unittest.main()

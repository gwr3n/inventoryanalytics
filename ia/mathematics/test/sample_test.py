import unittest
import ia.mathematics.arithmetics
import ia.mathematics.fibonacci

class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)
    
    def test_sum(self):
        self.assertEqual(ia.mathematics.arithmetics.sum(1,2),3)

    def test_fib(self):
        self.assertEqual(ia.mathematics.fibonacci.fib2(2),[1,1])

if __name__ == '__main__':
    unittest.main()
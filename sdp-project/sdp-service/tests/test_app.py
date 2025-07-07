import unittest
from sdp.app import app

class TestApp(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_predict_buggy(self):
        data_tuple = [559, 47, 70, 9, 2, 0.89, 3.0, 3, 2, 48, 13]
        data_tuple = [float(element) for element in data_tuple]
        response = self.client.post('http://127.0.0.1:5000/predict', json={'data_tuple': data_tuple})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['result'], [1])

    def test_predict_not_buggy(self):
        data_tuple = [154, 14, 34, 17, 2, 2.0, 4.5, 0, 0, 40, 10]
        data_tuple = [float(element) for element in data_tuple]
        response = self.client.post('http://127.0.0.1:5000/predict', json={'data_tuple': data_tuple})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['result'], [0])

if __name__ == '__main__':
    unittest.main()
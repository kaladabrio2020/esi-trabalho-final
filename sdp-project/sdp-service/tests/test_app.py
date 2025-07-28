import unittest
from src.sdp.app import app

class TestApp(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_predict_buggy(self):
        data_tuple = [
            3.0, 2, -0.13848717183784534, -0.1892451802249343, -0.9623843167356956,
            1.4933847662157607, 1963, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 1, 0, -0.007934346, -0.0017914104, -0.007667591,
            0.022907337, -0.0015613589
        ]
        response = self.client.post('/predict', json={'data_tuple': data_tuple})
        self.assertEqual(response.status_code, 200)
        predicted = response.get_json()['result'][0]
        self.assertIsInstance(predicted, int)
        self.assertGreaterEqual(predicted, 1)  # esperamos ao menos 1 bug

    def test_predict_not_buggy(self):
        data_tuple = [
            3.0, 1, -0.828975926576955, -0.19252659249677878, -0.5644251484191736,
            -0.6714132102320955, 1955, 2005, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
            0, 0, 0, 1, 0, 0, -0.010980263, -0.005899675, -0.0029756394,
            0.0061707986, -0.013852177
        ]
        response = self.client.post('/predict', json={'data_tuple': data_tuple})
        self.assertEqual(response.status_code, 200)
        predicted = response.get_json()['result'][0]
        self.assertIsInstance(predicted, int)
        self.assertGreaterEqual(predicted, 0)
        self.assertLessEqual(predicted, 20) 

if __name__ == '__main__':
    unittest.main()

import unittest
from src.sdp.app import app

class TestApp(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_predict_buggy(self):
        data_tuple = [3.0,1,-0.828975926576955,-0.19252659249677878,-0.5644251484191736,-0.6714132102320955,1955,2005,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,-0.010980263,-0.005899675,-0.0029756394,0.0061707986,-0.013852177]
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
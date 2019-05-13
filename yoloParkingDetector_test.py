import unittest
import yoloParkingDetector as ypd
from requests import get
from os.path import isfile

class TestParkingDetector(unittest.TestCase):

    # Set up parking detector object
    def setUp(self):
        lotID = "unitTest"
        camNum = 999
        vid = "lotVid.avi"
        yml = "unitTest.yml"

        classFile = "yolov3.txt"
        weightsFile = "yolov3.weights"
        configFile = "yolov3.cfg"

        self.detector = ypd.yoloParkingDetector(vid, classFile, weightsFile, configFile, yml, lotID, camNum)
        self.mockPostData = [{'id': '1', 'confidence': 0.8070341348648071}, {'id': '2', 'confidence': 0.9012327790260315}, {'id': '3', 'confidence': 0.9624365568161011},
                            {'id': '4', 'confidence': 0.8763743042945862}, {'id': '5', 'confidence': 0.8130772709846497}, {'id': '6', 'confidence': 0},
                            {'id': '7', 'confidence': 0.7185894846916199}, {'id': '8', 'confidence': 0.7005387544631958}, {'id': '9', 'confidence': 0.885391354560852},
                            {'id': '10', 'confidence': 0}, {'id': '11', 'confidence': 0.8569262623786926}, {'id': '12', 'confidence': 0.7939738631248474},
                            {'id': '13', 'confidence': 0.9334664344787598}, {'id': '14', 'confidence': 0}, {'id': '15', 'confidence': 0.9897783398628235},
                            {'id': '16', 'confidence': 0.922886848449707}, {'id': '17', 'confidence': 0.9869686365127563}, {'id': '18', 'confidence': 0.8296765089035034},
                            {'id': '19', 'confidence': 0}, {'id': '20', 'confidence': 0.8491337299346924}, {'id': '21', 'confidence': 0.6013709306716919},
                            {'id': '22', 'confidence': 0}, {'id': '23', 'confidence': 0.7269196510314941}, {'id': '24', 'confidence': 0.506303071975708},
                            {'id': '25', 'confidence': 0.7155311703681946}, {'id': '26', 'confidence': 0.6877471208572388}, {'id': '27', 'confidence': 0},
                            {'id': '28', 'confidence': 0}, {'id': '29', 'confidence': 0}, {'id': '30', 'confidence': 0}]

        self.mockPutData = [{'id': '1', 'confidence': 0.9}, {'id': '2', 'confidence': 0.9}, {'id': '3', 'confidence': 0.9},
                            {'id': '4', 'confidence': 0.9}, {'id': '5', 'confidence': 0.9}, {'id': '6', 'confidence': .9},
                            {'id': '7', 'confidence': 0.9}, {'id': '8', 'confidence': 0.6}, {'id': '9', 'confidence': 0.9},
                            {'id': '10', 'confidence': 0}, {'id': '11', 'confidence': 0.8569262623786926}, {'id': '12', 'confidence': 0.9},
                            {'id': '13', 'confidence': 0.9}, {'id': '14', 'confidence': 0}, {'id': '15', 'confidence': 0.9},
                            {'id': '16', 'confidence': 0.9}, {'id': '17', 'confidence': 0.9}, {'id': '18', 'confidence': 0.9},
                            {'id': '19', 'confidence': 0}, {'id': '20', 'confidence': 0.9}, {'id': '21', 'confidence': 0.9},
                            {'id': '22', 'confidence': 0}, {'id': '23', 'confidence': 0.9}, {'id': '24', 'confidence': 0.9},
                            {'id': '25', 'confidence': 0.9}, {'id': '26', 'confidence': 0.9}, {'id': '27', 'confidence': 0},
                            {'id': '28', 'confidence': 0}, {'id': '29', 'confidence': 0}, {'id': '30', 'confidence': 0.9}]


    def test_postValidStatus(self):
        self.detector.parkingStatus = self.mockPostData
        response1, response2 =  self.detector.postStatus()

        self.assertEqual(200, response1.status_code)
        self.assertEqual(200, response2.status_code)

        getReq = get(self.detector.URL + "/" + str(self.detector.parkinglot_ID))
        self.assertIn(str(self.mockPostData).replace("\'", "").replace(" ", ""), getReq.text.replace("\"", "").replace(" ", ""))

    def test_postInvalidStatus(self):
        self.detector.parkingStatus = ["invalid Data"]
        response1, response2 = self.detector.postStatus()

        #print(str(response1.status_code) + " " + response1.text + "\n" + str(response2.status_code) + " " + response2.text)

        self.assertEqual(422, response2.status_code)
        self.assertEqual(422, response1.status_code)


    def test_putValidStatus(self):
        self.detector.parkingStatus = self.mockPutData
        response = self.detector.putStatus()
        print(str(response.status_code) + " " + response.text)
        self.assertEqual(200, response.status_code)

    # Unexpected server crash with this test
    # def test_putInvalidStatus(self):
    #     self.detector.parkingStatus = ["invalid Data"]
    #     response = self.detector.putStatus()
    #
    #     print(str(response.status_code) + " " + response.text)
    #     self.assertEqual(422, response.status_code)

    def test_openYML(self):
        self.detector.openYML()
        self.assertIsNotNone(self.detector.parkingSpaceData)

    # Integration Test - test all components of yoloParkingDetector
    def test_run(self):
        self.detector.run()
        self.assertIsNotNone(self.detector.parkingSpaceData)
        self.assertIsNotNone(self.detector.net)

    def test_runInvalid(self):
        ypd.cap = None
        self.assertRaises(AttributeError, self.detector.run)


if __name__ == '__main__':
    unittest.main()

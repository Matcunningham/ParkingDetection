import base64
from requests import get
import json
ServerURL = "http://54.186.186.248:3000/api/overlayimage"
img = "test.jpg"

def getOverlay(filepath, parkinglotID):
    print("Sending overlay data request...")
    body = {'parkinglot_ID': parkinglotID}
    try:
        msgResponse = get(ServerURL, data=body)  # Post overlay
        payload = msgResponse.json()
        data = payload[0]['data']
        b64Data = data.encode('UTF-8')
        bytesData = base64.b64decode(b64Data)
        with open(filepath, "wb") as imageFile:
            imageFile.write(bytesData)
    except:
        raise ValueError('Status could not be posted')  # If post message fails catch error and print message


getOverlay(img, 491)
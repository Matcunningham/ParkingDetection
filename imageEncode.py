import base64
from requests import post
ServerURL = "http://54.186.186.248:3000/api/overlay"

def postOverlay(filepath, parkinglotID):
    with open(filepath, "rb") as imageFile:
        fileStr = base64.b64encode(imageFile.read())        #opens file and reads as string, converts to b64 encoding
    asciiImg = fileStr.decode('ascii')
    print("Sending overlay data...")
    body = {'parkinglot_id': parkinglotID, 'overlay': asciiImg}
    try:
        msgResponse = post(ServerURL, data=body)  # Post overlay
    except:
        raise ValueError('Status could not be posted')  # If post message fails catch error and print message

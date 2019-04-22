import base64
import yaml
from requests import post
ServerURL = "http://54.186.186.248:3000/api/overlaycoordinates"
coordinateFile = "test.yml"

def postOverlay(filepath, parkinglotID):
    # with open(filepath, "rb") as imageFile:
    #     fileBytes = base64.b64encode(imageFile.read())        #opens file and reads as string
    with open(filepath, 'r') as stream:
        yml = yaml.load(stream)
    print("Sending overlay coordinates...")
    body = {'parkinglot_ID': parkinglotID, 'data': str(yml)}
    try:
        msgResponse = post(ServerURL, data=body)  # Post overlay
    except:
        raise ValueError('Status could not be posted')  # If post message fails catch error and print message


postOverlay(coordinateFile, 112)
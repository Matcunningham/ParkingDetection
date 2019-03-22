import base64
from requests import post
ServerURL = "http://54.186.186.248:3000/api/overlayimage"
img = "smiley.jpg"

def postOverlay(filepath, parkinglotID):
    with open(filepath, "rb") as imageFile:
        fileBytes = base64.b64encode(imageFile.read())        #opens file and reads as string
    fileStr = fileBytes.decode('UTF-8')
    print("Sending overlay data...")
    body = {'parkinglot_ID': parkinglotID, 'data': fileStr}
    try:
        print(type(fileStr))
        print(fileStr)
        msgResponse = post(ServerURL, data=body)  # Post overlay
    except:
        raise ValueError('Status could not be posted')  # If post message fails catch error and print message


postOverlay(img, 491)
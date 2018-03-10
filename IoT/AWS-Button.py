import os
from urllib.parse import urlencode
 
from twilio.rest import Client
 
 
def handler(event, context):
    client = Client(
        os.environ['TWILIO_ACCOUNT'],
        os.environ['TWILIO_TOKEN'],
    )
 
    call = client.calls.create(
        to=os.environ['ALERT_PHONE'],
        from_=os.environ['TWILIO_PHONE'],
        url='http://twimlets.com/message?{}'.format(urlencode({
            'Message[0]': "Baby needs to go potty",
        })),
    )
 
    print(call.sid)

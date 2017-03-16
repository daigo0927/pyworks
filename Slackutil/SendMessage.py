import requests
import json

class SendMessage:

    def __init__(self,
                 webhook = 'webhook',
                 message = 'hello world!'):
        self.webhook = webhook
        self.message = message

    def send(self):
        message = {}
        message['text'] = self.message

        requests.post(self.webhook, data=json.dumps(message))
    

    
    

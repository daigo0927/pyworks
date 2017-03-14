import requests
import json

class SendMessage:

    def __init__(self,
                 webhook = 'https://hooks.slack.com/services/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',
                 message = 'hello world!'):
        self.webhook = webhook
        self.message = message

    def send(self):
        message = {}
        message['text'] = self.message

        requests.post(self.webhook, data=json.dumps(message))
    

    
    

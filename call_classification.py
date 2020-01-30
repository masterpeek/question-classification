import requests
import json

headers = {'content-type': 'application/json'}
URL = 'http://127.0.0.1:8000/api/v1.0/classification/'
data = {'text': 'คุณอายุเท่าใด'}

res = requests.post(URL, data = json.dumps(data), headers=headers)

result_classification = res.json().get('classification')
result_accuracy = res.json().get('accuracy')

print(result_classification, result_accuracy)
import requests
import json

headers = {'content-type': 'application/json'}
URL = 'http://127.0.0.1:8005/api/v1.0/deepcut/'
data = {'text': 'ท่านสนใจไปเที่ยวที่ไหน'}

res = requests.post(URL, data = json.dumps(data), headers=headers)

result_deepcut = res.json().get('result_deepcut')

print(result_deepcut)
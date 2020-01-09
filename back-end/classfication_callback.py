import boto3
import requests
import random
import time
from boto3.dynamodb.conditions import Key
import json
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table("xwwngz-db")

classfication_endpoint = "http://*.*.*.*:5001/classfication"
result_to_emoji = {
    0 : '👎🏻👎🏻👎🏻',
    1 : '👍👍👍'
}

def classfication_callback(text):
    table_id = random.randint(1, 1000000)
    requests.post(classfication_endpoint, data={"id":str(table_id), "text":text})
    timeout = time.time() + 60*2
    resp = "TIMEOUT"
    while True:
        resp = table.query(KeyConditionExpression=Key('id').eq(table_id))
        if len(resp['Items'])>0 or time.time() > timeout:
            break
        time.sleep(1)
    result = json.loads(resp['Items'][0]['result'])
    return result_to_emoji[result]
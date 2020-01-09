import os
import io
import boto3
import json
import csv

# grab environment variables
ENDPOINT_NAME = "gpt2-demo"
runtime= boto3.client('runtime.sagemaker')

def sagemaker_callback(text):
    payload = json.dumps({"data":text, "k": 3})
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='application/json',
                                       Body=payload)
    result = json.loads(response['Body'].read().decode())['sequences'].split('.;')
    return sorted(result, key=lambda x:len(x))[-1]

# print(sagemaker_callback("I love it"))
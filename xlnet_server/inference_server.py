from __future__ import absolute_import, division, print_function

import glob
import logging
import os
import random
import json
import torch
import time
import boto3
from transformers import WEIGHTS_NAME, XLNetForSequenceClassification, XLNetTokenizer
from flask import Flask, request
from train import infer
from config import checkpoint_dir, device

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table("xwwngz-db")
app = Flask(__name__)
model = None
tokenizer = None


def load_model():
    checkpoints = list(os.path.dirname(c) for c in sorted(
        glob.glob(checkpoint_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    model = XLNetForSequenceClassification.from_pretrained(checkpoints[0])
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    model.to(device)
    model.eval()
    return (model, tokenizer)


@app.route('/classfication', methods=['POST'])
def classfication_handler():
    data = request.form
    table_id = int(data['id'])
    text = data['text']
    classfication_result = infer([text], model, tokenizer)
    item = {
        'id' : table_id,
        'result' : str(classfication_result[0])
    }
    table.put_item(Item=item)
    return json.dumps({'code' : 200})


if __name__ == "__main__":
    model, tokenizer = load_model()
    app.run(host='0.0.0.0', port=5001)
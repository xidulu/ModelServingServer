from sagemaker_callback import sagemaker_callback
from classfication_callback import classfication_callback
from flask import Flask, request
import json

app = Flask(__name__)
apiPrefix = '/api/v1/'

@app.route(apiPrefix + 'send2backend', methods=['POST'])
def send2backend():
    data = json.loads(request.get_data(as_text=True))
    print(data)
    request_type = data['type']
    request_text = data['data']
    value_ = ""
    if request_type == 'generation':
        # In case of text generation, we invoke the SageMaker endpoint
        value_ = sagemaker_callback(request_text)
    else:
        value_ = classfication_callback(request_text)

    re = {
    'code': 0,
    'text':data,
    'result': value_,
    'message': "test"
    }
    return json.dumps(re)

if __name__=="__main__":
    app.run(debug=True, port=5001)

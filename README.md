# Model Serving Server

This repo is composed of three parts:
- A flask-based XLNet inference server.
- A frontend written in react-js.
- A "fake lambda" server that redirects frontend request to the correponding service.

## Model Training

To train the model, create `./xlnet_server/data/` and put your data under it, then run `python ./xlnet_server/train.py` to perform training.

To install dependency, run `conda install -c conda-forge transformers` and `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`. The model is trained using multiple GPUs with 256GB of memory in total.

After the training is finished, the model could be loaded as part of an inference server. To start the server, run `python inference_server`. The server is backboned by `Flask`.

## Front end

To start the frontend, enter `./front-end` and type `npm install`, `npm start`. The frontend webpage could be accessed via `http://localhost:3000`.

## Fake Lambda

The files under `back-end` are the so-called `fake lambda`, as it is designed to simulate the `Lambda` service in **AWS**, to start handling requests, run `python ./back-end/server.py`. Similar to the model server, the `fake lambda` is implemented with `Flask`.

---------------------

It should be noticed that, to reach the full functionality of this "system", some **AWS** services would be involved. The first one is a dynamoDB that stores the classfication result and the other one is a SageMaker GPT-2 endpoint responsible for text generation.
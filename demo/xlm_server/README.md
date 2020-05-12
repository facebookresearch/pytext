# TorchScript Server for XLM-R
In this directory, we provide a model server and a client console for DocNN and XLM-R text classifications models. These models were trained with PyTorch and exported to TorchScript


## Server
For monolingual DocNN,
```
$ mkdir build && cd build
$ # Copy the downloaded models into this directory
$ echo -e 'FROM pytext/predictor_service_torchscript:who\nCOPY *.torchscript /app/\nCMD ["./server","mono.model.pt.torchscript"]' >> Dockerfile
$ docker build -t server .
$ docker run -it -p 8080:8080 server
$ curl -d '{"text": "hi"}' -H 'Content-Type: application/json' localhost:8080
```
For multilingual XLM-R,
```
echo -e 'FROM pytext/predictor_service_torchscript:who\nCOPY *.torchscript /app/\nCMD ["./server","multi.model.pt.torchscript", "multi.vocab.model.pt.torchscript"]' >> Dockerfile
```


## Console
The console provides a front-end webpage to view the predictions of the classification model. It also allows for exploring model predictions interactively, and for logging corrections back to the server.

### Console Setup

```
$ python3 -m venv env
$ source env/bin/activate
$ (env) pip install -r requirements.txt
```

running the server:
```
$ python server.py --modelserver http://localhost:8080
```

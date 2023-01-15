# CPP Machine Learning Web Server

This is an example to deploy image classification model using [libasyik](https://github.com/okyfirmansyah/libasyik), a web services that runs on C++ and Boost libraries. Usually, [Flask](https://github.com/pallets/flask) or [Ray](https://github.com/ray-project/ray) are used to deploy machine learning model as a web service endpoint. This repository show how to use deploy machine learning model using C++ and expose the endpoint for use.

This example also shows how to integrate [Triton Inference Server](https://github.com/triton-inference-server/server) using its [C++ API Client](https://github.com/triton-inference-server/client). We can coupled the web services with [onnxruntime](https://github.com/microsoft/onnxruntime) to process the incoming data inside the proces. But I want to try the decoupled method to split the model inference process and web services that accept incoming data using REST API.

## Build Docker
```
// Build from Dockerfile
docker build -t cpp-ml-server:1.1.0 .

// Pull from Docker Registry
docker pull haritsahm/cpp-ml-server:1.1.0
```

## Run application

1. Sync submodules that have the model configurations
```
git submodule update --init --recursive
```

3. Git LFS on submodule
```
// Just in case the model isn't downloadade
cd triton-ml-server && git lfs pull && cd ..
```

4. Docker compose up
```
docker-compose up -d
```

## Examples
```python
import base64
import json

import cv2
import numpy as np
import requests
from PIL import Image

# Tree frog
url = "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01644373_tree_frog.JPEG?raw=true"

image = np.array(Image.open(requests.get(url, stream=True).raw))
image_string = base64.b64encode(cv2.imencode('.png', image)[1]).decode('utf-8')

response = requests.post("http://127.0.0.1:8080/classification/image", headers={"Content-Type":"application/json"}, data=json.dumps({"image":image_string}))
print(json.loads(response.text))
```

## TODO
- [ ] Add detailed data validation steps
- [ ] Optimize variables and parameters using pointers
- [ ] Support batched inputs
- [ ] Support coupled inference process using onnxruntime

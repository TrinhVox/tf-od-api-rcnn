FROM python:3.7

# install build utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get -y upgrade

# clone the repository 
RUN git clone --depth 1 https://github.com/tensorflow/models.git

# Install object detection api dependencies
RUN apt-get install -y protobuf-compiler python3-pil python3-lxml python3-tk && \
    pip install Cython && \
    pip install contextlib2 && \
    pip install jupyter && \
    pip install matplotlib && \
    pip install pycocotools && \
    pip install opencv-python && \
    pip install flask && \
    pip install tensorflow && \
    pip install Pillow && \
    pip install tf_slim && \
    pip install requests

# Get protoc 3.0.0, rather than the old version already in the container
RUN curl -OL "https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip" && \
    unzip protoc-3.0.0-linux-x86_64.zip -d proto3 && \
    mv proto3/bin/* /usr/local/bin && \
    mv proto3/include/* /usr/local/include && \
    rm -rf proto3 protoc-3.0.0-linux-x86_64.zip

# Run protoc on the object detection repo
RUN cd models/research && \
    protoc object_detection/protos/*.proto --python_out=.

# Set the PYTHONPATH to finish installing the API
ENV PYTHONPATH=$PYTHONPATH:/models/research/object_detection
ENV PYTHONPATH=$PYTHONPATH:/models/research/slim
ENV PYTHONPATH=$PYTHONPATH:/models/research

# clone the flask application
RUN git clone https://github.com/TrinhVox/tf-od-api-rcnn.git/

# set this as the working directory
WORKDIR /tf-od-api-rcnn

# download the pretrained model
# change here to download your pretrained model
RUN mkdir models && \
    cd models/ && \
    curl -O "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8.tar.gz" && \
    tar xzf "faster_rcnn_resnet152_v1_640x640_coco17_tpu-8.tar.gz" && \
    rm "faster_rcnn_resnet152_v1_640x640_coco17_tpu-8.tar.gz"

CMD ["python", "main.py"]

#FROM tensorflow/tensorflow:2.1.0-gpu-py3
FROM tensorflow/tensorflow:2.1.0-py3

# Set the working directory to /
WORKDIR /

# COPY the current directory contents into the container at /app
COPY pbt /pbt/
COPY requirements.txt /
COPY README.md /

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Run tensorboard and then main.py when the container launches
CMD ["sh", "-c", "tensorboard --logdir=/pbt/tmp/ --port 6006 --host 0.0.0.0", "&", "python3", "/pbt/main.py"]
#CMD ["python3", "/pbt/main.py"]

# Run with
# docker run -p 127.0.0.1:6006:6006 pbt:latest


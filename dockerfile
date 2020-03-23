#FROM tensorflow/tensorflow:2.1.0-gpu-py3
FROM tensorflow/tensorflow:2.1.0-py3

# Set the working directory to /
WORKDIR /

# COPY the current directory contents into the container at /app
COPY pbt /pbt/
COPY requirements.txt /
COPY README.md /

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements_docker.txt

# Run app.py when the container launches
#CMD ["sh", "-c", "tensorboard --logdir=/logs/ --port", "&&", "python3", "/run_training.py"]
CMD ["python3", "/pbt/main.py"]

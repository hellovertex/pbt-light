# tf-pbt
Framework for fast prototyping of Population-Based-Training methods. Implemented using Tensorflow 2.1.0 and tf_agents 0.4.0
## Using Docker (Tested on Ubuntu 18.04 LTS)
### Install Docker (quick and easy)
* Goto  https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-using-the-repository 
* After installation, create dockergroup and add user

   ```sudo groupadd docker```
* add user to group

   ```sudo usermod -aG docker $USER```
* restart (if it says authentication failed, the easiest fix is to restart computer)

   ```su -s $USER ```
* test if docker runs without sudo (OPEN NEW TERMINAL)

   ```docker run hello-world```

### Run
* Create Docker image (from inside project directory)

   ``` docker build -t pbt:latest . ```

* Run container (with attached port 6006 for tensorboard)

   ``` docker run -p 127.0.0.1:6006:6006 -it --name pbt pbt:latest bash ```

* Inside container, run tensorboard and python script

   ``` tensorboard --logdir /pbt/tmp/ --port 6006 --host 0.0.0.0 & python3 /pbt/main.py ```

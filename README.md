# mnist

## Build the container
First build our mnist docker container.
>docker build -t mnist:latest .

## Cassandra
Pull cassandra container from the Internet.
>docker pull cassandra:latest

Create a network.
>docker network create my-network

Connect the cassandra container to my network.
>docker run --name cassandra --net=my-network --net-alias=cassandra -p 9042:9042 -d cassandra:latest

## Connect dockers
Connect the mnist container to the same network.
>docker run --name mnist --net=my-network --net-alias=mnsit -d -p 8000:5000 mnist:latest


## Post your image to the service in the command line.
>curl -X post -F @image=the_path_of_your_image"http://localhost:5000/mnist"


# Image base

## environment

#### Docker build
```
docker build -t uwgmi-2022:v0 -f docker/Dockerfile ./
```

#### Docker run
```
docker run -it --rm --name image_base\
 --gpus all --shm-size=100g\
 -v $PWD:/workspace\
 -p 5555:8888 --ip=host\
 rsna-2022:v0
```
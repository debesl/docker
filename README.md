# Docker Images 
## How to use the images
### Dependencies:
  - Host has to be a Linux machine (manditory)
  - Visual Studio Code installed (recommended)

### Useful commands:
- docker build -f myDockerfile.Dockerfile . -t mydocker
- docker run mydocker [--runargs]
  - possible runargs: "--nework=host",
		    "--icp=host",
		    "--pid=host",
		    "--gpus",
		    "--all", and many more: https://docs.docker.com/reference/cli/docker/container/run/ 
  

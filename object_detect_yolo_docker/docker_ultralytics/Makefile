.PHONY: build up down rm logs gui-up gui-run

# Nome da imagem e do container
IMAGE_NAME = ultralytics-jetson-opencv
CONTAINER_NAME = ultra_jetson
DEVICE = /dev/video1

# Build da imagem local
build:
	docker build -t $(IMAGE_NAME) .

# Rodar container com imagem local e GUI
gui-up: build
	xhost +local:root
	docker run -it \
		--runtime=nvidia \
		--privileged \
		--ipc=host \
		--device=$(DEVICE):$(DEVICE) \
		-e DISPLAY=$(DISPLAY) \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME)

# Rodar container diretamente da imagem da Ultralytics
gui-run:
	xhost +local:docker
	t=ultralytics/ultralytics:latest-jetson-jetpack4 && \
	sudo docker run -it --rm \
		--ipc=host \
		--runtime=nvidia \
		--gpus all \
		-v /home/$$USER:/home/user \
		--device=/dev/video1:/dev/video1 \
		-e DISPLAY=$$DISPLAY \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		$$t

# Parar o container (caso esteja rodando sem --rm)
down:
	docker stop $(CONTAINER_NAME)

# Remover container parado
rm:
	docker rm $(CONTAINER_NAME)

# Ver logs (caso rode sem --rm)
logs:
	docker logs -f $(CONTAINER_NAME)


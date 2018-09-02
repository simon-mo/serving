# This Makefile is for development purpose
all: build run
build:
	docker build -t clipper-tf-serving/inception-v3 -f InceptionV3Dockerfile .
run:
	NV_GPU=1 docker run -d \
		--runtime=nvidia \
		-e CUDA_VISIBLE_DEVICES=1 \
		-p 8500:8500 \
		-t clipper-tf-serving/inception-v3
run-long-wait:
	NV_GPU=1 docker run -d \
		--runtime=nvidia \
		-e CUDA_VISIBLE_DEVICES=1 \
		-p 8500:8500 \
		-e max_batch_size=50 \
		-e batch_timeout_micros=2000000 \
		-t clipper-tf-serving/inception-v3
clean:
	docker kill `docker ps -q`


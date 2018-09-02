# This file is for dev purpose
build:
	bazel build -c opt tensorflow_serving/example:inception_client_cc
run:
	bazel-bin/tensorflow_serving/example/inception_client_cc --batch_size 10  --server_port="127.0.0.1:8500" --image_file="/home/ubuntu/cat.jpeg" --model_name="inception" --model_signature_name="predict_images"

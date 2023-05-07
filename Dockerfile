FROM public.ecr.aws/lambda/python:3.9

# Install packages
RUN python3.9 -m pip install python-dotenv onnxruntime "transformers[torch]" --target "${LAMBDA_TASK_ROOT}"

# FROM 454099489730.dkr.ecr.eu-west-2.amazonaws.com/drslimms-ecr-registry:onnx-function-amd64

# Set Production Environment
ENV ENV=prod

# Create directory for app and onnx
WORKDIR /var/task
COPY app.py .
COPY onnx ./onnx

# Set TRANSFORMERS_CACHE
RUN mkdir /tmp/transformers
ENV TRANSFORMERS_CACHE=/tmp/transformers

CMD [ "app.lambda_handler" ]

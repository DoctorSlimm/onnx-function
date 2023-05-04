FROM public.ecr.aws/lambda/python:3.9


# Install the runtime interface client
RUN python3.9 -m pip install --target . awslambdaric
RUN python3.9 -m pip install python-dotenv onnxruntime "transformers[torch]"

# Set Production Environment
ENV ENV=prod

# Copy files
COPY app.py ./

# Copy onnx directory
COPY onnx onnx


# Set Up Entrypoints
COPY ./entry_script.sh /entry_script.sh
ADD aws-lambda-rie-arm64 /usr/local/bin/aws-lambda-rie-arm64
ENTRYPOINT ["/entry_script.sh"]
CMD [ "app.lambda_handler" ]


# Node https://github.com/aws/aws-lambda-nodejs-runtime-interface-client
# RUN npm install aws-lambda-ric
#ENTRYPOINT ["/usr/local/bin/npx", "aws-lambda-ric"]
#CMD ["app.handler"]

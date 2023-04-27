#!/bin/sh
if [ -z "${AWS_LAMBDA_RUNTIME_API}" ]; then
  exec /usr/local/bin/aws-lambda-rie-arm64 python3.9 -m awslambdaric app.lambda_handler
else
  exec python3.9 -m awslambdaric app.lambda_handler
fi

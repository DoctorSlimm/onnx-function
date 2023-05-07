import traceback
from time import time
import numpy as np
from dotenv import load_dotenv
from onnxruntime import InferenceSession
from transformers import AutoTokenizer

load_dotenv()

# Worth Investigating
# https://blog.ml6.eu/the-art-of-pooling-embeddings-c56575114cf8
# https://github.com/UKPLab/sentence-transformers/issues/46#issuecomment-1152816277

tokenizer = AutoTokenizer.from_pretrained('./onnx')
session = InferenceSession("onnx/model.onnx")


def lambda_handler(event, context):
    try:
        if 'ping' in event:
            print('Pinging')
            t0 = time()
            return {
                'total_time': time() - t0,
            }
        if 'modelInputs' in event:
            print('Inference\n')
            model_inputs = event['modelInputs']
            text = model_inputs['text']

            ##############################
            # Batches
            ##############################
            if not isinstance(text, list):
                encoded_inputs = tokenizer(text, return_tensors="np")
                model_outputs = session.run(None, input_feed=dict(encoded_inputs))[0]
                token_embeddings = model_outputs  # Single (1, 11, 768), Batch (batch_size, sequence_length, hidden_size)
                special_token_ids = [
                    tokenizer.cls_token_id,
                    tokenizer.unk_token_id,
                    tokenizer.sep_token_id,
                    tokenizer.pad_token_id,
                    tokenizer.mask_token_id,
                ]

                # Mask to exclude special tokens from pooling calculation
                mask = np.ones(token_embeddings.shape[:-1], dtype=bool)

                # Mean Pooling Sentence Embedding
                for special_token_id in special_token_ids:
                    mask &= encoded_inputs['input_ids'] != special_token_id  # Exclude special tokens from mask
                mean_pooled_embeddings = np.sum(token_embeddings * mask[..., np.newaxis], axis=1)  # Apply mask and take sum over sequence dimension
                mean_pooled_embeddings = np.mean(mean_pooled_embeddings, axis=0)  # Take mean over batch dimension

                return {
                    'embeddings': [*mean_pooled_embeddings.tolist()],
                }

            else:
                encoded_inputs = tokenizer.batch_encode_plus(text, return_tensors="np", padding=True, truncation=True, max_length=512)
                model_outputs = session.run(None, input_feed=dict(encoded_inputs))[0]
                token_embeddings = model_outputs  # Single (1, 11, 768), Batch (batch_size, sequence_length, hidden_size)
                special_token_ids = [
                    tokenizer.cls_token_id,
                    tokenizer.unk_token_id,
                    tokenizer.sep_token_id,
                    tokenizer.pad_token_id,
                    tokenizer.mask_token_id,
                ]

                mask = np.ones(token_embeddings.shape[:-1], dtype=bool)

                # Mean Pooling Sentence Embedding
                mean_pooled_embeddings = np.sum(token_embeddings * mask[..., np.newaxis], axis=1)  # Apply mask and take sum over sequence dimension

                return {
                    'embeddings': [*mean_pooled_embeddings.tolist()]
                }

    except Exception as e:
        print(e)
        return {
            'error': str(traceback.format_exc()) + str(e)
        }


if __name__ == "__main__":
    import requests
    e = {
        "modelInputs": {
           "text": "Burning tonnes of Oil to prove my manhood"
        }
    }
    r = lambda_handler(e, None)
    # r = requests.post('http://localhost:9000/2015-03-31/functions/function/invocation', json=e)

    embs = r['embeddings']
    arr = np.array(embs)
    print(arr.shape)

    e = {
        "modelInputs": {
            "text": [
                "Burning tonnes of Oil to prove my manhood",
                "China is a communist country",
                "I love my country",
            ],
        }
    }
    r = lambda_handler(e, None)

    embs = r['embeddings']
    arr = np.array(embs)
    print('Batched')
    print(arr.shape)


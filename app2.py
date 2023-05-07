import traceback
from time import time
import numpy as np
from dotenv import load_dotenv
from onnxruntime import InferenceSession
from transformers import AutoTokenizer

load_dotenv()

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

            if isinstance(text, list):
                encoded_inputs = tokenizer.batch_encode_plus(text, return_tensors="np", padding=True)
            else:
                encoded_inputs = tokenizer(text, return_tensors="np")

            model_outputs = session.run(None, input_feed={
                'input_ids': encoded_inputs['input_ids'],
                'attention_mask': encoded_inputs['attention_mask'],
                'token_type_ids': encoded_inputs['token_type_ids'],
            })
            token_embeddings = model_outputs[0]  # (batch_size, sequence_length, hidden_size)

            # Mask to exclude special tokens from pooling calculation
            mask = np.ones(token_embeddings.shape[:-1], dtype=bool)
            special_token_ids = [
                tokenizer.cls_token_id,
                tokenizer.unk_token_id,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id,
                tokenizer.mask_token_id,
            ]
            for special_token_id in special_token_ids:
                mask &= encoded_inputs['input_ids'] != special_token_id

            # Max Pooling Sentence Embedding
            max_pooled_embeddings = np.max(token_embeddings * mask[..., np.newaxis], axis=1)
            max_pooled_embeddings = np.mean(max_pooled_embeddings, axis=0)

            # Mean Pooling Sentence Embedding
            mean_pooled_embeddings = np.sum(token_embeddings * mask[..., np.newaxis], axis=1)
            mean_pooled_embeddings = np.sum(mean_pooled_embeddings, axis=1) / np.sum(mask, axis=1)
            mean_pooled_embeddings = np.mean(mean_pooled_embeddings, axis=0)

            return {
                'modelOutputs': {
                    # 'raw': model_outputs.tolist(),
                    'token_embeddings': token_embeddings.tolist(),
                    'max_pooled_embeddings': max_pooled_embeddings.tolist(),
                    'mean_pooled_embeddings': mean_pooled_embeddings.tolist(),
                }
            }

    except Exception as e:
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

    embs = r['modelOutputs']['mean_pooled_embeddings']
    arr = np.array(embs)
    print('Single')

    assert arr.shape == (768,)

    e = {
        "modelInputs": {
            "text": [
                "Burning tonnes of Oil to prove my manhood",
                "Burning tonnes of Oil to prove my manhood",
                "Burning tonnes of Oil to prove my manhood",
            ],
        }
    }
    r = lambda_handler(e, None)

    embs = r['modelOutputs']['mean_pooled_embeddings']
    arr = np.array(embs)
    print('Batched')

    assert arr.shape == (3, 768)


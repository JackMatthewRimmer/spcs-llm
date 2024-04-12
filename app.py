from flask import Flask, request, jsonify
from pydantic import BaseModel, model_validator, ValidationError
from typing import List, Dict, Any
from transformers import LlamaTokenizer, LlamaForCausalLM 
import os

SERVICE_HOST = os.getenv('SERVICE_HOST', '0.0.0.0')
SERVICE_PORT = os.getenv('SERVICE_PORT', '8080')
app = Flask(__name__)

@app.get('/healthcheck')
def healthcheck():
    return 'STATUS: READY'

'''
This endpoint is our entry point for generating a chat completion.
We are expecting this incoming JSON to be in the format:
{
    "data": {
        [
            [0, "systemPrompt", "userPrompt", {"inputValues": "someValue"}, {"metadata": "someMetadata"}],
        ]
    }
} 
'''
@app.post('/complete')
def complete():
    try:
        body: CompletionRequest = CompletionRequest.model_validate_json(request.data) 
    except ValidationError as e:
        return f"Serialization failed with error: {e}"
    print(f"Request: {body}")

    print("Loading model...")
    model, tokenizer = load_model()
    print("Model loaded")

    prompt: str = f'''
    <s>[INST] <<SYS>>
    { body.systemPrompt }
    <</SYS>>

    { body.userPrompt } [/INST]
    '''

    print(f"Generating chat completion for prompt: {prompt}")
    print("Encoding prompt...")
    encoded = tokenizer.encode(prompt, return_tensors='pt')
    print("Prompt encoded")
    print("Generating chat completion...")
    output = model.generate(encoded)
    print("Chat completion generated")
    print("Decoding chat completion...")
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated chat completion: {decoded}")


def load_model():    
    print("Loading model...")
    model = LlamaForCausalLM.from_pretrained('models/')
    print("Loading tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained('models/')
    return model, tokenizer

class CompletionRequest(BaseModel):
    systemPrompt: str
    userPrompt: str
    inputValues: Dict[str, Any]
    metadata: Dict[str, Any]

    @model_validator(mode='before')
    def array_to_object(cls, values):

        data_array: List[Any] = values["data"]

        if data_array is None or len(data_array) == 0:
            return ValidationError(f"data array must contain exactly one row: {values}")

        if len(data_array) > 1:
            return ValidationError(f"More than one row provided in request {values}")

        return {
            "systemPrompt": data_array[0][1],
            "userPrompt": data_array[0][2],
            "inputValues": data_array[0][3],
            "metadata": data_array[0][4],
        }

if __name__ == '__main__':
    app.run(SERVICE_HOST, SERVICE_PORT, debug=True)
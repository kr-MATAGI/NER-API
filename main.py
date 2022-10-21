from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import torch
from transformers import ElectraTokenizer
from ner_api import load_ner_api, make_response_json
from electra_lstm_crf import ELECTRA_POS_LSTM

class Req_Sent(BaseModel):
    id: str
    text: str

class Req_Data(BaseModel):
    date: str
    sentences: List[Req_Sent]


### APP
app = FastAPI()

model_path = "./model/model.pth"
tokenizer_name = "monologg/koelectra-base-v3-discriminator"
tokenizer = ElectraTokenizer.from_pretrained(tokenizer_name)
model = torch.load(model_path)

@app.get("/")
async def root():
    return {"msg": "Hello World"}

@app.post("/ner")
async def response_ner(item: Req_Data):
    response_list = []
    for sent_idx, sent in enumerate(item.sentences):
        model_outputs = load_ner_api(model, tokenizer, input_sent=sent.text)
        model_outputs.id = sent.id
        response_list.append(model_outputs)
    res_json_str = make_response_json(model_output_list=response_list)
    print(f"=======================\nRequest: \n{item}")
    print(f"=======================\nResponse: \n{res_json_str}\n=======================")
    return res_json_str
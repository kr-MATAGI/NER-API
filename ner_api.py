import json
import datetime as dt
import copy
import numpy as np

import torch
from transformers import ElectraTokenizer
from electra_lstm_crf import ELECTRA_POS_LSTM

from definition.tag_def import ETRI_TAG, MECAB_POS_TAG
from definition.data_def import (
    Res_results, Res_ne,
    Mecab_Item, Tok_Pos
)

#### ids2tag / tag2ids
pos_tag2ids = {v: int(k) for k, v in MECAB_POS_TAG.items()}
pos_ids2tag = {k: v for k, v in MECAB_POS_TAG.items()}
ne_ids2tag = {v: k for k, v in ETRI_TAG.items()}

#===========================================================================
def make_response_json(model_output_list):
#===========================================================================
    root_json_dict = {
        "date": "",
        "results": []
    }
    result_json_dict = {
        "id": "",
        "text": "",
        "ne": []
    }

    root_json_dict["date"] = dt.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    for data in model_output_list:
        result_json_dict["id"] = data.id
        result_json_dict["text"] = data.text
        for ne_idx, ne_item in enumerate(data.ne_list):
            res_ne = {
                "id": ne_item.id, "word": ne_item.word, "label": ne_item.label,
                "begin": ne_item.begin, "end": ne_item.end
            }
            result_json_dict["ne"].append(res_ne)
        root_json_dict["results"].append(copy.deepcopy(result_json_dict))
    json_string = json.dumps(root_json_dict, ensure_ascii=False)
    return json_string

#===========================================================================
def tokenize_mecab_pair(mecab_pair_list, tokenizer):
#===========================================================================
    # [ (word, [(word, [pos,...]), ...] ]
    # -> [ (word, [(tokens, [pos,...]), ...] ]

    tokenized_mecab_list = []
    for m_idx, mecab_pair in enumerate(mecab_pair_list):
        new_pos_pair_list = []
        for m_pos_idx, m_pos_pair in enumerate(mecab_pair[1]):
            tokenized_word = tokenizer.tokenize(m_pos_pair[0])
            new_pos_pair_list.append(Tok_Pos(tokens=tokenized_word,
                                             pos=m_pos_pair[1]))
        tokenized_mecab_list.append(Mecab_Item(word=mecab_pair[0], tok_pos_list=new_pos_pair_list))

    return tokenized_mecab_list

#===========================================================================
def make_pos_tag_ids(src_sent, mecab, tokenizer, input_ids, max_pos_len: int = 3):
#===========================================================================
    split_sent = src_sent.split()
    word_pair_list = [] # [ (word, (word,pos))... ]
    for word_idx, word_item in enumerate(split_sent):
        mecab_pos_res = mecab.pos(word_item)
        conv_list = []
        for pos_res in mecab_pos_res:
            conv_list.append((pos_res[0], pos_res[1].split("+")))
        word_pair_list.append((word_item, conv_list))

    mecab_token_pos_list = tokenize_mecab_pair(word_pair_list, tokenizer)

    tok_pos_item_list = []
    for mecab_item in mecab_token_pos_list:
        tok_pos_item_list.extend(mecab_item.tok_pos_list)

    # pos_ids
    pos_ids = []
    for tp_idx, tok_pos in enumerate(tok_pos_item_list):
        conv_pos = []
        filter_pos = [x if "UNKNOWN" != x and "NA" != x and "UNA" != x and "VSV" != x else "O" for x in tok_pos.pos]
        conv_pos.extend([pos_tag2ids[x] for x in filter_pos])
        if max_pos_len > len(conv_pos):
            diff_len = (max_pos_len - len(conv_pos))
            conv_pos += [pos_tag2ids["O"]] * diff_len
        if max_pos_len < len(conv_pos):
            conv_pos = conv_pos[:max_pos_len]
        pos_ids.append(conv_pos)

        if 1 < len(tok_pos.tokens):
            for _ in range(len(tok_pos.tokens) - 1):
                empty_pos = [pos_tag2ids["O"]] * max_pos_len
                pos_ids.append(empty_pos)

    input_ids_len = len(input_ids)
    pos_ids.insert(0, [pos_tag2ids["O"]] * max_pos_len)  # [CLS]
    if input_ids_len <= len(pos_ids):
        pos_ids = pos_ids[:input_ids_len - 1]
        pos_ids.append([pos_tag2ids["O"]] * max_pos_len)  # [SEP]
    else:
        pos_ids_size = len(pos_ids)
        for _ in range(input_ids_len - pos_ids_size):
            pos_ids.append([pos_tag2ids["O"]] * max_pos_len)

    return pos_ids
#===========================================================================
def load_ner_api(model, tokenizer, input_sent):
#===========================================================================
    token_res = tokenizer(input_sent)
    # pos_tag_ids = make_pos_tag_ids(src_sent=input_sent, mecab=mecab,
    #                                tokenizer=tokenizer, input_ids=token_res["input_ids"])
    # pos_tag_ids = np.array(pos_tag_ids)
    inputs = {
        "input_ids": torch.LongTensor([token_res["input_ids"]]),
        "attention_mask": torch.LongTensor([token_res["attention_mask"]]),
        "token_type_ids": torch.LongTensor([token_res["token_type_ids"]]),
        # "pos_tag_ids": torch.LongTensor([pos_tag_ids])
    }

    outputs = model(**inputs)
    res_ids_seq = outputs[0]

    # Extract NE Items
    conv_label_seq = [ne_ids2tag[x] for x in res_ids_seq]
    conv_label_seq = conv_label_seq[1:-1]
    input_ids = token_res["input_ids"][1:-1]
    # conv_pos_ids = pos_tag_ids[1:-1]
    # giho_pos_ids = [35, 36, 37, 38, 39, 40]

    ne_items_list = []  # (id, NE, Label)
    concat_tok_list = []
    curr_label = ""
    for idx, (ids, label) in enumerate(zip(input_ids, conv_label_seq)):
        # print(tokenizer.convert_ids_to_tokens(ids), label)
        if "B-" in label:
            if 0 < len(concat_tok_list):
                add_str = "".join(concat_tok_list).replace("##", "")
                ne_items_list.append((len(ne_items_list) + 1, add_str, curr_label))
                concat_tok_list.clear()
                curr_label = ""
            curr_label = label.replace("B-", "")
            conv_tok = tokenizer.convert_ids_to_tokens(ids)
            if 0 < len(concat_tok_list) and "##" not in conv_tok:
                concat_tok_list.append(" ")
            concat_tok_list.append(conv_tok)
        elif "I-" in label:
            conv_tok = tokenizer.convert_ids_to_tokens(ids)
            if 0 < len(concat_tok_list) and "##" not in conv_tok:
                concat_tok_list.append(" ")
            concat_tok_list.append(conv_tok)
        else:
            if 0 < len(concat_tok_list):
                add_str = "".join(concat_tok_list).replace("##", "")
                ne_items_list.append((len(ne_items_list) + 1, add_str, curr_label))
                concat_tok_list.clear()
                curr_label = ""

    # Make Json Response
    # print(ne_items_list)
    last_find_position = 0
    res_results_data = Res_results(text=input_sent)
    for ne_item in ne_items_list:
        ne_begin = input_sent.find(ne_item[1], last_find_position)
        if -1 == ne_begin:
            continue
        ne_end = ne_begin + len(ne_item[1]) - 1
        last_find_position = ne_end
        res_ne = Res_ne(id=str(ne_item[0]), word=ne_item[1], label=ne_item[2],
                        begin=ne_begin, end=ne_end)
        res_results_data.ne_list.append(res_ne)

    return res_results_data

#### MAIN ####
if "__main__" == __name__:
    '''
        id: str = ""
        word: str = ""
        label: str = ""
        begin: int = ""
        end: int = ""
    '''
    is_need_model_save = False
    if is_need_model_save:
        model = ELECTRA_POS_LSTM.from_pretrained("./model")
        print(model)
        torch.save(model, "./model/model.pth")
        exit()

    is_need_local_test = False
    if is_need_local_test:
        model_path = "./model/model.pth"
        tokenizer_name = "monologg/koelectra-base-v3-discriminator"
        for i in load_ner_api(model_path=model_path, tokenizer_name=tokenizer_name):
            print(i)
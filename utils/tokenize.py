# $ wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
# $ pip install en_vectors_web_lg-2.1.0.tar.gz
import en_vectors_web_lg
import re
import numpy as np
import os
import pickle
from transformers import BertTokenizer, BertModel
import torch

def clean(w):
    return re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            w.decode().lower()
            ).replace('-', ' ').replace('/', ' ')


def tokenize(key_to_word):
    key_to_sentence = {}
    for k, v in key_to_word.items():
        key_to_sentence[k] = [clean(w[0]) for w in v if clean(w[0]) != '']
    return key_to_sentence

def getEmbeddings(text, model, tokenizer):
  marked_text = "[CLS] " + text + " [SEP]"
  tokenized_text = tokenizer.tokenize(marked_text)
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  segments_ids = [1] * len(tokenized_text)

  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensors = torch.tensor([segments_ids])

  with torch.no_grad():
    outputs = model(tokens_tensor, segments_tensors)
    hidden_states = outputs[2]
    # token_embeddings = torch.stack(hidden_states, dim=0)
    # token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # token_embeddings = token_embeddings.permute(1,0,2)

    token_vecs = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)

    return sentence_embedding.numpy()


def create_dict(key_to_sentence, dataroot, use_glove=True):
    token_file = dataroot+"/token_to_ix.pkl"
    glove_file = dataroot+"/train_glove.npy"
    bert_file = dataroot+"/train_bert.npy"
    if use_glove and os.path.exists(glove_file) and os.path.exists(token_file):
        print("Loading glove train language files")
        return pickle.load(open(token_file, "rb")), np.load(glove_file)
    elif (not use_glove) and os.path.exists(bert_file) and os.path.exists(token_file):
        print("Loading bert train language files")
        return pickle.load(open(token_file, "rb")), np.load(bert_file)

    print("Creating train language files")
    token_to_ix = {
        'UNK': 1,
    }

    spacy_tool = None
    pretrained_emb = []
    if use_glove:
        spacy_tool = en_vectors_web_lg.load()
        pretrained_emb.append(spacy_tool('UNK').vector)
        print ("Using glove embedding")
    else:
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
        pretrained_emb.append(getEmbeddings('UNK', bert_model, bert_tokenizer))
        print ("Using bert embedding")


    for k, v in key_to_sentence.items():
        for word in v:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)
                if use_glove:
                    pretrained_emb.append(spacy_tool(word).vector)
                else:
                    pretrained_emb.append(getEmbeddings(word, bert_model, bert_tokenizer))

    pretrained_emb = np.array(pretrained_emb)
    if use_glove:
        np.save(glove_file, pretrained_emb)
    else:
        np.save(bert_file, pretrained_emb)
    pickle.dump(token_to_ix, open(token_file, "wb"))
    return token_to_ix, pretrained_emb

def sent_to_ix(s, token_to_ix, max_token=100):
    ques_ix = np.zeros(max_token, np.int64)

    for ix, word in enumerate(s):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix + 1 == max_token:
            break

    return ques_ix


def cmumosei_7(a):
    if a < -2:
        res = 0
    if -2 <= a and a < -1:
        res = 1
    if -1 <= a and a < 0:
        res = 2
    if 0 <= a and a <= 0:
        res = 3
    if 0 < a and a <= 1:
        res = 4
    if 1 < a and a <= 2:
        res = 5
    if a > 2:
        res = 6
    return res

def cmumosei_2(a):
    if a < 0:
        return 0
    if a >= 0:
        return 1

def pad_feature(feat, max_len):
    if feat.shape[0] > max_len:
        feat = feat[:max_len]

    feat = np.pad(
        feat,
        ((0, max_len - feat.shape[0]), (0, 0)),
        mode='constant',
        constant_values=0
    )

    return feat

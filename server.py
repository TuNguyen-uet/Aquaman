from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from transformers import AutoModel, AutoTokenizer
from pyvi import ViTokenizer, ViPosTagger

import torch
import numpy as np
import pandas as pd

app = Flask(__name__)

aspectMebe = ['ship', 'giá', 'chính hãng', 'chất lượng', 'dịch vụ', 'an toàn', 'other']
aspectTech = ['cấu hình', 'mẫu mã', 'hiệu năng', 'ship', 'giá', 'chính hãng', 'dịch vụ', 'phụ kiện', 'other']
numAspectMebe = 6
numAspectTech = 8

def preprocess(input):
    acronyms = pd.read_csv('data/vocab/acronym.txt', sep=',', header=None, names=["acronym", "meaning"])
    stopwords = pd.read_csv('data/vocab/vietnamese_stopwords.txt', sep=',', header=None, names=["stopword"])

    texts = input.split(' ')
    ans = []
    for text in texts:
        # remove duplicate last letter
        while len(text) > 1 and text[-1] == text[-2]:
            text = text[:-1]
        # remove too long or null word
        if len(text) > 7 or len(text) < 1:
            continue
        ans.append(text)
    input = " ".join(ans)

    word_paths = ViPosTagger.postagging(ViTokenizer.tokenize(input))[0]
    word_paths_copy = [x.lower() for x in word_paths]
    word_paths = word_paths_copy
    input = " ".join(word_paths)

    texts = input.split(' ')
    ans = []
    for text in texts:
        # replace acronym with corresponding word
        if text in acronyms["acronym"].values:
            text = acronyms[acronyms["acronym"] == text].iloc[0, 1]
        # remove stopword
        if text in stopwords.values:
            continue

        ans.append(text)
    input = " ".join(ans)
    return input

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    aspectName = aspectMebe
    numAspect = numAspectMebe

    userInput = [preprocess(x) for x in request.form.values()]

    phobert = AutoModel.from_pretrained("vinai/phobert-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    bertOutput = []
    input = userInput[0]
    input_ids = torch.tensor([tokenizer.encode(input)])
    with torch.no_grad():
        bertOutput.append(phobert(input_ids).pooler_output.numpy()[0])
    bertOutput = np.array(bertOutput)
    predicts = []

    for i in range(numAspect):
        pretrainedModel = load_model('model/bert_mlp_mebe_shopee_{}'.format(i))
        predict = pretrainedModel.predict_classes(bertOutput)
        if (predict[0][0] != 0):
            predicts.append(aspectName[i])

    return render_template('index.html', input_text = "Text: {}".format(userInput[0]),
                           prediction_text=predicts)

@app.route('/predictBertCNN',methods=['POST'])
def predictBertCNN():
    aspectName = aspectMebe
    numAspect = numAspectMebe

    userInput = [x for x in request.form.values()]

    phobert = AutoModel.from_pretrained("vinai/phobert-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    bertOutput = []
    input = userInput[0]
    input_ids = torch.tensor([tokenizer.encode(input)])

    with torch.no_grad():
        feature = (phobert(input_ids).last_hidden_state)
        size = feature.shape[1]
        input_tensor = torch.zeros(35, 768)
        index_tensor = torch.ones(size, 768, dtype=torch.int64)
        for i in range(size):
            index_tensor[i] *= (35 - size + i)
        dim = 0
        input_tensor.scatter_(dim, index_tensor, feature[0])
        feature = input_tensor.numpy()
        bertOutput.append(feature)
    bertOutput = np.array(bertOutput)

    predicts = ""
    for i in range(numAspect):
        pretrainedModel = load_model('model/bert_cnn_mebe_shopee_{}'.format(i))
        predict = pretrainedModel.predict_classes(bertOutput)
        if (predict[0][0] != 0):
            predicts = predicts + ", " + aspectName[i]

    return render_template('index.html', input_text = "Text: {}".format(userInput),
                           prediction_text='Aspects: {}'.format(predicts))

if __name__ == "__main__":
    app.run(debug=False)
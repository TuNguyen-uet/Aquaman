import  pandas as pd
from models import  Input, AspectOutput, PolarityOutput
import string


def isNan(string):
    return string != string

def contains_digit(w):
    for i in w:
        if i.isdigit():
            return True
    return False

def typo_trash_labeled(lst):
    for i in lst:
        if i in useless_labels:
            return True
    return False

punctuations = list(string.punctuation)
useless_labels = [296, 315, 330, 349, 295, 314, 329, 348]

def load_data(path, NUM_OF_ASPECTS):
    """

    :param path:
    :return:
    :rtype: list of models.Input
    """
    if NUM_OF_ASPECTS == 6:
        aspect_name = ['ship', 'giá', 'chính hãng', 'chất lượng', 'dịch vụ', 'an toàn']
    else:
        aspect_name = ['cấu hình','mẫu mã','hiệu năng','ship','giá','chính hãng','dịch vụ','phụ kiện','other']
    inputs = []
    outputs = []
    df = pd.read_csv(path, encoding='utf8')

    for index, row in df.iterrows():
        if isNan(row['text']) == 0:
            text = row['text'].strip()
            inputs.append(Input(text))

            labels = list(range(NUM_OF_ASPECTS))
            scores = [0 if row[aspect_name[i]] == 0 else 1 for i in range(0, NUM_OF_ASPECTS)]
            outputs.append(AspectOutput(labels, scores))

    return inputs, outputs

def get_labels(path):
    df = pd.read_csv(path)
    labels = []
    for i in range(len(df)):
        anno = (df['annotations'])[i]
        if isNan(anno) == 0 and isNan((df['text'])[i]) == 0:
            labels_list = []
            while True:
                anno_temp = anno.partition('\'label\':')[2].strip()
                if len(anno_temp) == 0:
                    break
                else:
                    labels_list.append(int(anno_temp[:3]))
                    anno = anno_temp
            labels.append(labels_list)
    return labels

def preprocess_inputs(inputs):
    """

       :param list of models.Input inputs:
       :return:
       :rtype: list of models.Input
       """
    _corpus = []
    for i in range(len(inputs)):
        t = inputs[i].text
        t = (t.strip()).split(' ')
        for w in t:
            if w in punctuations:
                t.remove(w)
        for j in range(len(t)):
            if contains_digit(t[j]):
                t[j] = '0'
        t = ' '.join(t)
        inputs[i].text = t
        _corpus.append(t)
    return inputs, _corpus

def make_corpus(_corpus, _labels):
    corpus = []
    labels = []
    for i in range(len(_corpus)):
        if typo_trash_labeled(_labels[i]) == 0:
            corpus.append(_corpus[i])
            labels.append(_labels[i])

    return corpus, labels

def make_vocab(corpus):
    vocab = []
    for text in corpus:
        text = text.split(' ')
        for w in text:
            if w not in vocab:
                vocab.append(w)

    with open('D:\DS&KT Lab\\NCKH\Aquaman_Deep_Project\Aquaman\data_mebe\mebe_shopee_vocab.txt', 'w', encoding='utf8') as f:
    # with open('D:\DS&KT Lab\\NCKH\Aquaman_Deep_Project\Aquaman\data_mebe\mebe_tiki_vocab.txt', 'w', encoding='utf8') as f:
    # with open('D:\DS&KT Lab\\NCKH\Aquaman_Deep_Project\Aquaman\data_tech\\tech_shopee_vocab.txt', 'w', encoding='utf8') as f:
    # with open('D:\DS&KT Lab\\NCKH\Aquaman_Deep_Project\Aquaman\data_tech\\tech_tiki_vocab.txt', 'w', encoding='utf8') as f:
        for w in vocab:
            f.write('{}\n'.format(w))

    return vocab

def get_aspect_scores(outputs, _labels):
    aspect_scores = []
    for i in range(len(outputs)):
        if typo_trash_labeled(_labels[i]) == 0:
            aspect_scores.append(outputs[i].scores)

    return aspect_scores

def preprocess_tiki(inputs):
    """

    :param list of models.Input inputs:
    :return:
    :rtype: list of models.Input
    """
    pass


def preprocess_dulich(inputs):
    """

    :param list of models.Input inputs:
    :return:
    :rtype: list of models.Input
    """
    pass

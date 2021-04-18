import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer


aspect_name = ['ship', 'giá', 'chính hãng', 'chất lượng', 'dịch vụ', 'an toàn']
# aspect_name = ['cấu hình', 'mẫu mã', 'hiệu năng', 'ship', 'giá', 'chính hãng', 'dịch vụ', 'phụ kiện']

def get_feature(id, corpus, scores):
    cv = CountVectorizer()
    X = cv.fit_transform(corpus)

    y = np.array([output[id] for output in scores])
    skb = SelectKBest(chi2, k='all')
    _chi2 = skb.fit_transform(X, y)

    feature_names = cv.get_feature_names()
    _chi2_scores = skb.scores_
    _chi2_pvalues = skb.pvalues_

    fout = {}
    fout['word'] = feature_names
    fout['scores'] = list(_chi2_scores)
    fout['pvalues'] = list(_chi2_pvalues)

    df = pd.DataFrame(fout, index=None, columns=['word', 'scores', 'pvalues'])
    df = df.sort_values('scores', ascending=False)
    mean_score = df['scores'].mean()
    mean_pvalue = df['pvalues'].mean()

    with open('D:\DS&KT Lab\\NCKH\Aquaman_Deep_Project\Aquaman\data_mebe\chi2\\shopee\chi2_shopee_{}.txt'.format(aspect_name[id]), 'w', encoding='utf8') as f:
    # with open('D:\DS&KT Lab\\NCKH\Aquaman_Deep_Project\Aquaman\data_mebe\mebe_tiki_vocab.txt', 'w', encoding='utf8') as f:
    # with open('D:\DS&KT Lab\\NCKH\Aquaman_Deep_Project\Aquaman\data_tech\\tech_shopee_vocab.txt', 'w', encoding='utf8') as f:
    # with open('D:\DS&KT Lab\\NCKH\Aquaman_Deep_Project\Aquaman\data_tech\\tech_tiki_vocab.txt', 'w', encoding='utf8') as f:
        for w,s,p in zip(df['word'],df['scores'],df['pvalues']):
            if s >= mean_score and p <= mean_pvalue:
                f.write('{}  {}  {}\n'.format(w,s,p))

    file = open('D:\DS&KT Lab\\NCKH\Aquaman_Deep_Project\Aquaman\data_mebe\chi2\\shopee\chi2_shopee_{}.txt'.format(aspect_name[id]), 'r',encoding='utf8')
    # file = open('D:\DS&KT Lab\\NCKH\Aquaman_Deep_Project\Aquaman\data_mebe\chi2\\tiki\chi2_tiki_{}.txt'.format(aspect_name[id]), 'r', encoding='utf8')
    # file = open('D:\DS&KT Lab\\NCKH\Aquaman_Deep_Project\Aquaman\data_tech\chi2\\shopee\chi2_shopee_{}.txt'.format(aspect_name[id]), 'r', encoding='utf8')
    # file = open('D:\DS&KT Lab\\NCKH\Aquaman_Deep_Project\Aquaman\data_tech\chi2\\tiki\chi2_tiki_{}.txt'.format(aspect_name[id]), 'r', encoding='utf8')
    texts = []
    while (True):
        line = file.readline().strip()
        if len(line) > 1:
            texts.append(line)
        if not line:
            break
    file.close()

    mlp_vocab = []
    mlp_vocab_score = []
    for text in texts:
        t = text.split(' ')
        mlp_vocab.append(t[0])
        mlp_vocab_score.append(float(t[2]))
    print(len(mlp_vocab))
    input_mlp = []
    for text in corpus:
        t = text.split(' ')
        text_vec = np.zeros(len(mlp_vocab))
        for w in t:
            if w in mlp_vocab:
                id = mlp_vocab.index(w)
                text_vec[id] = mlp_vocab_score[id]
        input_mlp.append(text_vec)
    input_mlp = np.array(input_mlp)

    return input_mlp, y



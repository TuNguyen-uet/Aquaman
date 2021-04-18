from sklearn.model_selection import train_test_split

from modules.preprocess import load_data, get_labels, preprocess_inputs, make_corpus, make_vocab, get_aspect_scores
from modules.evaluate import cal_aspect_prf
from modules.aspect.MLP_model import MLP_Model
from feature_extraction import get_feature


if __name__ == '__main__':
                                            # LOAD AND PREPROCESS THE DATA
    # Load data
    path_csv = 'D:\DS&KT Lab\\NCKH\Aquaman_Deep_Project\Aquaman\data_mebe\mebe_shopee.csv'
    # path_csv = 'D:\DS&KT Lab\\NCKH\Aquaman_Deep_Project\Aquaman\data_mebe\mebe_tiki.csv'
    # path_csv = 'D:\DS&KT Lab\\NCKH\Aquaman_Deep_Project\Aquaman\data_tech\\tech_shopee.csv'
    # path_csv = 'D:\DS&KT Lab\\NCKH\Aquaman_Deep_Project\Aquaman\data_tech\\tech_tiki.csv'
    NUM_OF_ASPECTS = 6
    # NUM_OF_ASPECTS = 8
    inputs, outputs = load_data(path_csv, NUM_OF_ASPECTS)   #3006, 3006

    # Get labels of all texts
    _labels = get_labels(path_csv)    #3006

    # Remove punctuation, digits in inputs
    # Make a _corpus of all texts in inputs
    inputs, _corpus = preprocess_inputs(inputs)  #3006

    # Make a corpus, a labels list of texts from input which aren't labeled typo/trash
    corpus, labels = make_corpus(_corpus, _labels)            #2086, 2086

    # Make a vocabulary from all words in corpus
    vocab = make_vocab(corpus)              #20362

    # Get scores of all texts in corpus
    scores = get_aspect_scores(outputs, _labels)     #(2086,7)

    if NUM_OF_ASPECTS == 6:
        aspect_name = ['ship', 'giá', 'chính hãng', 'chất lượng', 'dịch vụ', 'an toàn']
    else:
        aspect_name = ['cấu hình','mẫu mã','hiệu năng','ship','giá','chính hãng','dịch vụ','phụ kiện']
    model_predicts = []
    y_test = []

                                            # RUN THE MODELS FOR EVERY ASPECT
    for i in range(NUM_OF_ASPECTS):
        # Create inputs for MLP model
        input_mlp, y = get_feature(i, corpus, scores)

        x_tr, x_te, y_tr, y_te = train_test_split(input_mlp, y, test_size=0.2, random_state=20)

        # Create a model
        model = MLP_Model(NUM_OF_ASPECTS)

        #Train and predict model with sklearn Classifier
        model.train_sklearn(i, x_tr, y_tr)
        predict = model.predict_sklearn(i, x_te, y_te)

        # # Train and predict model with keras
        # model.train_keras(i, x_tr, y_tr)
        # predict = model.predict_keras(i, x_te, y_te)

        model_predicts.append(predict)
        y_test.append(y_te)

                                            # NORMALIZE THE PREDICTIONS TO THE RIGHT FORM
    for i in range(NUM_OF_ASPECTS):
        model_predicts[i] = list(model_predicts[i])
        y_test[i] = list(y_test[i])

    _predicts = []
    _y_tests = []
    for pre, te in zip(model_predicts, y_test):
        predict = []
        test = []
        for i, j in zip(pre, te):
            predict.append([i])
            test.append([j])
        _predicts.append(predict)
        _y_tests.append(test)

    predicts = []
    y_tests = []
    for i in range(len(_predicts[0])):
        pre = []
        for _pre in _predicts:
            pre.append(_pre[i][0])
        te = []
        for _te in _y_tests:
            te.append(int(_te[i][0]))
        predicts.append(pre)
        y_tests.append(te)

                                            # PRINT OUT THE RESULTS
    print('\t\t\tship\t\t\tgiá\t\t\tchính hãng\tchất lượng\t\tdịch vụ\t\tan toàn')
    # print('\tcấu hình\t\tmẫu mã\t\thiệu năng\tship\t\tgiá\t\tchính hãng\t\tdịch vụ\tphụ kiện')
    cal_aspect_prf(y_tests, predicts, num_of_aspect=NUM_OF_ASPECTS, verbal=True)
from modules.models import Model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class MLP_Model(Model):
    def __init__(self, NUM_OF_ASPECT):
        if NUM_OF_ASPECT == 6:
            self.NUM_OF_ASPECTS = NUM_OF_ASPECT
            self.aspect_name = ['ship', 'giá', 'chính hãng', 'chất lượng', 'dịch vụ', 'an toàn']
        else:
            self.NUM_OF_ASPECTS = 8
            self.aspect_name = ['cấu hình', 'mẫu mã', 'hiệu năng', 'ship', 'giá', 'chính hãng', 'dịch vụ', 'phụ kiện']

        self.model_sklearn_Classifier = MLPClassifier()
        self.model_keras_Sequential = Sequential()

    def train_sklearn(self, id, inputs, outputs):
        print('Training {}: ...'.format(self.aspect_name[id]))
        print(inputs.shape, inputs)
        clf = self.model_sklearn_Classifier.fit(inputs, outputs)

    def predict_sklearn(self, id, x_test, y_test):
        predict = self.model_sklearn_Classifier.predict(x_test)
        # print('Accuracy score:', metrics.accuracy_score(y_test, predict), '\n',
        #       'Precision score: ', metrics.precision_score(predict, y_test, zero_division='warn', average='weighted'), '\n',
        #       'Recall score: ', metrics.recall_score(predict, y_test, zero_division='warn', average='weighted'))
        print('Result board {}:'.format(self.aspect_name[id]))
        print(classification_report(y_test, predict))
        return predict

    def train_keras(self, id, inputs, outputs):
        print('Training {}: ...'.format(self.aspect_name[id]))
        self.model_keras_Sequential.add(Dense(64, activation='relu'))
        self.model_keras_Sequential.add(Dense(1, activation='sigmoid'))

        self.model_keras_Sequential.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        self.model_keras_Sequential.fit(inputs, outputs, epochs=20, batch_size=100)
        self.model_keras_Sequential.summary()

    def predict_keras(self, id, x_test, y_test):
        predict = self.model_keras_Sequential.predict_classes(x_test)
        print('Result board {}:'.format(self.aspect_name[id]))
        print(classification_report(y_test, predict))

        return predict

    def save(self, path):
        pass

    def load(self, path):
        pass

    def _represent(self, inputs):
        pass
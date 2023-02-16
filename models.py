import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense


class Model:

    def __init__(self):
        self.name = ''
        path = 'dataset/kidney_disease.csv'
        df = pd.read_csv(path)
        df = df[['sg', 'al', 'rc', 'sc', 'su', 'dm', 'hemo', 'pcv', 'htn', 'classification']]
        self.split_data(df)

    def split_data(self, df):
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Scale feature data
        x_scaler = MinMaxScaler()
        x = x_scaler.fit_transform(x)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def mlp_model(self):
        self.name = 'MLP Classifier'
        classifier = Sequential()
        classifier.add(Dense(256, input_dim=len(self.x_train[0]), kernel_initializer='random_normal', activation='relu'))
        classifier.add(Dense(1, activation='hard_sigmoid'))
        classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = classifier.fit(self.x_train, self.y_train, epochs=200, batch_size=self.x_train.shape[0])
        return classifier

    def accuracy(self, model):
        _, accuracy = model.evaluate(self.x_test, self.y_test)
        print("{self.name} has accuracy of {accuracy * 100} % ")


if __name__ == '__main__':
    model = Model()
    model.accuracy(model.mlp_model())

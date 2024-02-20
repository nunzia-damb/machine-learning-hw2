import os
from collections import Counter
from os.path import join
import matplotlib.pyplot as plt
import skf
import sklearn.metrics
import numpy as np
from PIL import Image
from keras import Model, Input
from keras.src import regularizers
from keras.src.callbacks import History
from keras.src.layers import Conv2D, Flatten, Dense, Dropout, MaxPool2D
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay, \
    multilabel_confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight


def create_model(input_shape, num_classes, base_width=16, depth=2):
    inputs = Input(input_shape)

    image_size = input_shape[0]
    filters = base_width
    kernel_size = 3
    # feature extractor
    for i in range(depth):
        if i == 0:
            x = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation="relu",
                       kernel_regularizer=regularizers.l2(0.001),
                       strides=1,
                       padding="same")(inputs)
        else:
            x = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation="relu",
                       kernel_regularizer=regularizers.l2(0.001),
                       strides=1,
                       padding="same")(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2),  padding="valid")(x)
        filters *= 2
        x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)

    outputs = Dense(num_classes, activation="softmax", kernel_regularizer=regularizers.l2(0.001))(x)
    model = Model(inputs, outputs, name="first_CNN_model")
    # an independent loss for each model
    #losses = ["categorical_crossentropy"]
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001 ), metrics=['accuracy'])

    return model

#unbalanced. Use f1-score or roc curve
def load_data_train(dir):
    X,Y = [], []
    for i in range(5):
        path = join(dir,str(i))
        for img in os.listdir(path):
            #Y.append([i])
            Y.append(i)
            path_img = join(path, img)
            img = np.asarray(Image.open(path_img))  # returns numpy array with size (96,96,3) with values between[0,1]
            X.append(img)
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape, Y.shape, Counter(Y))

    #oversampling
    #sm = SMOTE(random_state=42)
    #X_res, y_res = sm.fit_resample(X, Y)
    #print(X_res.shape, Counter(y_res))

    #normalization
    X_res = X / 255.0

    input_shape=(X_res.shape[1], X_res.shape[2], X_res.shape[3])
    num_classes = np.max(np.unique(Y)) + 1
    y_res = to_categorical(Y, num_classes)
    print(X_res.shape, y_res.shape, num_classes, input_shape)
    return X_res, y_res, input_shape, num_classes

def load_data_test(dir):
    X, Y, y = [], [], []
    for i in range(5):
        path = join(dir, str(i))
        for img in os.listdir(path):
            Y.append([i])
            path_img = join(path, img)
            img =np.asarray(Image.open(path_img))
            X.append(img)
    X = np.array(X)
    #normalize
    X = X/255.0
    input_shape = (X.shape[1], X.shape[2], X.shape[3])
    num_classes = np.max(np.unique(Y)) + 1
    Y = to_categorical(Y, num_classes)
    return X, Y, input_shape, num_classes

train_data_dir = "public/train/"
test_data_dir = "public/test/"

X_train, Y_train, input_shape, num_classes = load_data_test(train_data_dir)
X_test, Y_test,_,_= load_data_test(test_data_dir)

print(len(X_train), len(Y_train), X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

model = create_model(input_shape, num_classes)
model.summary()

history = History()
class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(Y_train, axis=1)), y=np.argmax(Y_train, axis=1))
class_weight_dict = dict(enumerate(class_weights))

history = model.fit(X_train, Y_train, batch_size=32, epochs=5, validation_data=(X_test, Y_test), shuffle=True, class_weight=class_weight_dict)


print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#score = accuracy_score(np.argmax(Y_test, axis=-1), np.argmax(prediction, axis=-1))
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
Y_test = np.argmax(Y_test, axis=1)
target_names = ['class1', 'class2','class3','class4','class5']
print(classification_report(Y_test, y_pred, target_names= target_names))
cm = multilabel_confusion_matrix(Y_test, y_pred)
#print(cm)
cm_display = ConfusionMatrixDisplay.from_predictions(y_true=Y_test, y_pred=y_pred)
plt.show()




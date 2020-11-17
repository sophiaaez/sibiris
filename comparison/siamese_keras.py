import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE

import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, GaussianNoise, Input, Conv2D, Dropout, Reshape, Flatten, Activation, add, concatenate
from tensorflow.keras.metrics import Recall, Accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.losses import cosine_similarity, binary_crossentropy, log_cosh

def find_indices(ids, amount=10):
    counter = 0
    indices = list()
    for i in range(amount):
        indices.append(i)
        name = ids[counter][-1]
        
        amount = 0
        for j in range(len(ids)):
            if i != j and name == ids[j][-1]:
                indices.append(j)
                amount += 1
                if amount == 10:
                    break
        counter += 1
    return indices

test_ids  = np.load('../ae/ae_test_encoding_ids.npy')
train_ids = np.load('../ae/ae_training_encoding_ids.npy')
test_enc  = np.load('../ae/ae_test_encodings.npy')
train_enc = np.load('../ae/ae_training_encodings.npy')
total_ids = np.concatenate([test_ids, train_ids], axis=0)
total_enc = np.concatenate([test_enc, train_enc], axis=0)

scaler = MinMaxScaler()
train_enc = scaler.fit_transform(train_enc)
test_enc = scaler.transform(test_enc)
total_enc = scaler.transform(total_enc)

real = Input(shape=(16384,))
fake = Input(shape=(16384,))

image_input = Input(shape=(8, 8, 256))
noise = GaussianNoise(0.2)(image_input)
layer1 = Conv2D(256, (3, 3), strides=(1, 1), activation="relu", padding="same")(noise)
layer2 = Conv2D(256, (3, 3), strides=(2, 2), activation="relu", padding="same")(layer1)
layer3 = Conv2D(256, (3, 3), strides=(2, 2), activation="linear", padding="same")(noise)
layer  = Activation("relu")(add([layer2, layer3]))
layer  = BatchNormalization()(layer)
layer1 = Conv2D(256, (3, 3), strides=(1, 1), activation="relu", padding="same")(layer)
layer2 = Conv2D(256, (3, 3), strides=(2, 2), activation="linear", padding="same")(layer1)
layer3 = Conv2D(256, (3, 3), strides=(2, 2), activation="linear", padding="same")(layer)
layer  = Activation("relu")(add([layer2, layer3]))
layer  = BatchNormalization()(layer)
layer1 = Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same")(layer)
layer2 = Conv2D(512, (3, 3), strides=(2, 2), activation="linear", padding="same")(layer1)
layer3 = Conv2D(512, (3, 3), strides=(2, 2), activation="linear", padding="same")(layer)
layer  = Activation("relu")(add([layer2, layer3]))
layer  = BatchNormalization()(layer)
output = Conv2D(1024, (1, 1), strides=(1, 1), activation="linear", padding="same")(layer)
output = Flatten()(output)
output = Dense(512, activation="relu")(output)
conv_net = Model(image_input, output, name="ConvNet")
print(conv_net.summary())

real_layer = Reshape((8, 8, 256))(real)
real_layer = conv_net(real_layer)

fake_layer = Reshape((8, 8, 256))(fake)
fake_layer = conv_net(fake_layer)

layer = concatenate([real_layer, fake_layer], axis=-1)
layer = Dense(128, activation="relu")(layer)
layer = BatchNormalization()(layer)
layer = Dropout(0.25)(layer)

layer = Dense( 32, activation="relu")(layer)
layer = BatchNormalization()(layer)
output = Dense( 1, activation="sigmoid")(layer)

model = Model([real, fake], output)
# model.add_loss(tf.reduce_sum(log_cosh(real_layer, fake_layer)))
model.compile(optimizer=Adam(lr=5e-4), loss='mse', metrics=['acc', Recall(thresholds=[0.2, 0.8])])

batch_size = 64
def train_generator(enc, ids):
    def find_same(id):
        for i in range(len(ids)):
            if ids[i, -1] == id:
                return i
        return -1

    count = len(enc)
    while True:
        encodings1 = np.empty((batch_size, 16384))
        encodings2 = np.empty((batch_size, 16384))
        output = np.zeros((batch_size, 1))

        for i in range(batch_size):
            index1 = np.random.randint(0, count)
            if i % 3 == 0:
                index2 = find_same(ids[index1, -1])
            else:
                index2 = np.random.randint(0, count)
   
            encodings1[i] = enc[index1]
            encodings2[i] = enc[index2]
            output[i] = 1 if ids[index1, -1] == ids[index2, -1] else 0
        yield [encodings1, encodings2], output


def test_generator(enc, ids, total_enc, total_ids):
    def find_same(id):
        for i in range(len(ids)):
            if total_ids[i, -1] == id:
                return i
        return None

    count = len(enc)
    total_count = len(total_ids)
    while True:
        encodings1 = np.empty((batch_size, 16384))
        encodings2 = np.empty((batch_size, 16384))
        output = np.zeros((batch_size, 1))

        for i in range(batch_size):
            index1 = np.random.randint(0, count)
            if i % 3 == 0:
                index2 = find_same(ids[index1, -1])
            else:
                index2 = np.random.randint(0, total_count)
   
            encodings1[i] = enc[index1]
            encodings2[i] = total_enc[index2]
            output[i] = 1 if total_ids[index1, -1] == total_ids[index2, -1] else 0
        yield [encodings1, encodings2], output

model.fit_generator(
    train_generator(train_enc, train_ids),
    steps_per_epoch=10 * len(train_enc) // batch_size,
    epochs=100,
    validation_data=test_generator(test_enc, test_ids, total_enc, total_ids),
    validation_steps=10 * len(test_enc) // batch_size
)

from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical

# モデルの定義
model = Sequential()

# --------------------------------------------------------------
# ここを埋めてください
model.add(Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=(2, 2), strides=(1,1), padding="same"))#畳み込み層　パラメーターについては後述
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"))#プーリング層
model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding="same")) #畳み込み層 パラメーターについては後述
model.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1)))#プーリング層
#コンパイルしてニューラルネットワークモデルの生成
#model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
# --------------------------------------------------------------
model.add(Flatten())
model.add(Dense(256)) #全結合層
model.add(Activation('sigmoid'))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()
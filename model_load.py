import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from dataloader import DataLoader



dataloader = DataLoader()
data, labels = dataloader.loadData()

        model= keras.models.Sequential()

model.add(keras.layers.Conv2D(150,(3,3),input_shape=data.shape[1:],activation=tf.nn.relu))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(keras.layers.Conv2D(100,(3,3),activation=tf.nn.relu))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(keras.layers.Conv2D(75,(3,3),activation=tf.nn.relu))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(45,activation='relu'))
model.add(keras.layers.Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

(train_data, test_data, train_labels, test_labels) = train_test_split(data,labels,test_size=0.1)

model.fit(train_data, train_labels,epochs=20)

#model.save('D:\Downloads\mask_model')
print(model.evaluate(test_data, test_labels))

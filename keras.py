# %%
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# %%
(X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()
X_train.shape

# %%
X_test.shape

# %%
X_train[0]

# %%
plt.imshow(X_train[0])

# %%
plt.figure(figsize=(15,2))
plt.imshow(X_train[1])

# %%
y_train = y_train.reshape(-1,)
y_train[:5]

# %%
y_test = y_test.reshape(-1,)

# %%
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# %%
def plot_sample(X,y,index):
    plt.figure(figsize=(15,2))
    plt.imshow(X_train[index])
    plt.xlabel(classes[y[index]])

# %%
plot_sample(X_train, y_train, 0)

# %%
plot_sample(X_train, y_train, 0)

# %%
# ANN MODEL
ann = models.Sequential([
        layers.Flatten(input_shape=(32,32,3)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='softmax')    
    ])

ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=2)


# %%
ann.evaluate(X_test,y_test)

# %%
# from sklearn.metrics import confusion_matrix , classification_report
# import numpy as np
# y_pred = ann.predict(X_test)
# y_pred_classes = [np.argmax(element) for element in y_pred]

# print("Classification Report: \n", classification_report(y_test, y_pred_classes))

# %%
# cnn model
cnn = models.Sequential([
    
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# %%
cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %%
cnn.fit( X_train,y_train,epochs=10)

# %%
cnn.evaluate(X_test,y_test)

# %%
# y_test
y_test = y_test.reshape(-1,)
y_test[:5]


# %%
plot_sample(X_test,y_test,1)

# %%
y_pred = cnn.predict(X_test)
y_pred[:10]

# %%
y_classes = [np.argmax(element) for element in y_pred]
y_classes[:10]

# %%
y_test[:10]

# %%
# Print the class names for the first 10 predictions
for i in range(10):
    plot_sample(X_test, y_test,i)
    print(classes[y_classes[i]])

# %%




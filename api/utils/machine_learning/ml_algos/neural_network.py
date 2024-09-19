import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

# https://www.askpython.com/python/examples/top-machine-learning-algorithms
# https://www.askpython.com/python/examples/neural-networks
# https://www.askpython.com/python/examples/single-perceptron-neural-network


def test_neural_net():
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels) , (test_images, test_labels) = mnist.load_data()

    # Printing the shapes
    print("train_images shape: ", train_images.shape)
    print("train_labels shape: ", train_labels.shape)
    print("test_images shape: ", test_images.shape)
    print("test_labels shape: ", test_labels.shape)

    # Displaying first 9 images of dataset
    fig = plt.figure(figsize=(10, 10))

    nrows = 3
    ncols = 3
    for i in range(9):
        fig.add_subplot(nrows, ncols, i + 1)
        plt.imshow(train_images[i])
        plt.title("Digit: {}".format(train_labels[i]))
        plt.axis(False)
    plt.show()

    # Converting image pixel values to 0 - 1
    train_images = train_images / 255
    test_images = test_images / 255

    print("First Label before conversion:")
    print(train_labels[0])

    # Converting labels to one-hot encoded vectors
    train_labels = tf.keras.utils.to_categorical(train_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels)

    print("First Label after conversion:")
    print(train_labels[0])

    # Using Sequential() to build layers one after another
    model = tf.keras.Sequential([

        # Flatten Layer that converts images to 1D array
        tf.keras.layers.Flatten(),

        # Hidden Layer with 512 units and relu activation
        tf.keras.layers.Dense(units=512, activation='relu'),

        # Output Layer with 10 units for 10 classes and softmax activation
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])

    model.compile(
      loss = 'categorical_crossentropy',
      optimizer = 'adam',
      metrics = ['accuracy']
    )

    history = model.fit(
      x = train_images,
      y = train_labels,
      epochs = 10
    )

    # Showing plot for loss
    plt.plot(history.history['loss'])
    plt.xlabel('epochs')
    plt.legend(['loss'])
    plt.show()

    # Showing plot for accuracy
    plt.plot(history.history['accuracy'], color='orange')
    plt.xlabel('epochs')
    plt.legend(['accuracy'])
    plt.show()

    # Call evaluate to find the accuracy on test images
    test_loss, test_accuracy = model.evaluate(
        x=test_images,
        y=test_labels
    )

    print("Test Loss: %.4f" % test_loss)
    print("Test Accuracy: %.4f" % test_accuracy)

    predicted_probabilities = model.predict(test_images)
    predicted_classes = tf.argmax(predicted_probabilities, axis=-1).numpy()

    index = 11

    # Showing image
    plt.imshow(test_images[index])

    # Printing Probabilities
    print("Probabilities predicted for image at index", index)
    print(predicted_probabilities[index])

    print()

    # Printing Predicted Class
    print("Probabilities class for image at index", index)
    print(predicted_classes[index])


def single_perceptron():
    # Dataset
    df = pd.DataFrame()
    df['x'] = [random.randint(1, 20) for x in range(10)]
    df['y'] = [random.randint(1, 20) for x in range(10)]
    df.head()

    label = []
    for i in range(df.shape[0]):
        if df.iloc[i, 0] < df.iloc[i, 1]:
            label.append(1)
        else:
            label.append(-1)

    df['label'] = label
    print(df)

    # weights and bias
    weights = [np.round(random.uniform(-0.99, 0.99), 2) for i in range(2)]

    w = weights.copy()
    X = [[df.iloc[i,0],df.iloc[i,1]] for i in range(df.shape[0])]
    wx = [X[i][0]*w[0]+X[i][1]*w[1] for i in range(df.shape[0])]

    # Signum activation function
    for i in range(df.shape[0]):
        if wx[i]<=0:
            pred = 0
        else:
            pred = 1

    for i in range(df.shape[0]):
        print('wx : ', wx[i])
        if wx[i] <= 0:
            pred = -1
        else:
            pred = 1
        print('label=', df['label'][i])
        print('pred = ', pred)
        if pred != df['label'][i]:
            err = df['label'][i] - pred
            print('err', err)
            print('before', w[0], w[1])
            w[0] = w[0] + err
            w[1] = w[1] + err
            print('after', w[0], w[1])
        else:
            print('w_i', w[0], w[1])

    # Test Dataset
    new_df = pd.DataFrame()
    new_df['x'] = [random.randint(1, 20) for x in range(100)]
    new_df['y'] = [random.randint(1, 20) for x in range(100)]
    new_df.head()

    label_text = []
    for i in range(new_df.shape[0]):
        if new_df.iloc[i, 0] < new_df.iloc[i, 1]:
            label_text.append(1)
        else:
            label_text.append(-1)
    new_wX = w[0] * new_df['x'] + w[1] * new_df['y']

    new_df['given label'] = label_text
    pred_label_text = []

    for i in range(new_df.shape[0]):
        if new_wX[i] >= 0:
            pred_label_text.append(-1)
        else:
            pred_label_text.append(1)

    new_df['predicted labels'] = pred_label_text





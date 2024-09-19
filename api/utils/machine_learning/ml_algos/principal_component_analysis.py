import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import cv2
import os

# https://www.askpython.com/python/examples/top-machine-learning-algorithms
# https://www.askpython.com/python/examples/principal-component-analysis-for-image-data


# Import the dataset
def import_dataset():
    digits = load_digits()
    df = digits.data
    print(df.shape)
    return df, digits


# View sample image (remember image is in the form of numpy array)
def view_data(df):
    image_sample = df[0,:].reshape(8,8)
    plt.imshow(image_sample)


"""
1. Reduce Image Dimensions

Now, using PCA, let’s reduce the image dimensions from 64 to just 2 so that we can visualize the
dataset using a Scatterplot.

sklearn provides us with a very simple implementation of PCA.

We’ll use the sklearn.decomposition provides PCA() class to implement principal component analysis algorithm.

It accepts integer number as an input argument depicting the number of principal components we want
in the converted dataset.

We can also pass a float value less than 1 instead of an integer number. i.e. PCA(0.90) this means
the algorithm will find the principal components which explain 90% of the variance in data.
"""
def reduce_image_dimensions(digits):
    pca = PCA(2)  # we need 2 principal components.
    converted_data = pca.fit_transform(digits.data)
    print(converted_data.shape)
    return converted_data


def visualize_results(converted_data, digits):
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(10, 6))
    c_map = plt.cm.get_cmap('jet', 10)
    plt.scatter(converted_data[:, 0], converted_data[:, 1], s=15, cmap=c_map, c=digits.target)
    plt.colorbar()
    plt.xlabel('PC-1'), plt.ylabel('PC-2')
    plt.show()


def load_image(image_filename):
    # image_filename = 'my_doggo_sample.jpg'
    img = cv2.imread(image_filename)  # you can use any image you want.
    plt.imshow(img)
    # cv2.imshow(image_filename, img)
    return img


def split_image_RGB(img):
    # Splitting the image in R,G,B arrays.
    blue, green, red = cv2.split(img)
    # it will split the original image into Blue, Green and Red arrays.
    return red, green, blue


def apply_pca_RGB(red, green, blue, n_components=20):
    # initialize PCA with first n_components principal components
    pca = PCA(n_components)

    # Applying to red channel and then applying inverse transform to transformed array.
    red_transformed = pca.fit_transform(red)
    red_inverted = pca.inverse_transform(red_transformed)

    # Applying to Green channel and then applying inverse transform to transformed array.
    green_transformed = pca.fit_transform(green)
    green_inverted = pca.inverse_transform(green_transformed)

    # Applying to Blue channel and then applying inverse transform to transformed array.
    blue_transformed = pca.fit_transform(blue)
    blue_inverted = pca.inverse_transform(blue_transformed)

    return red_inverted, green_inverted, blue_inverted


"""
4. Compressing the Image

Inverse Transformation is necessary to recreate the original dimensions of the base image.

In the process of reconstructing the original dimensions from the reduced dimensions, some information is lost as we keep only selected principal components, 20 in this case.

Stacking the inverted arrays using dstack function. Here it is important to specify the datatype of our arrays, as most images are of 8 bit. Each pixel is represented by one 8-bit byte.
"""


def compress(red_inverted, green_inverted, blue_inverted):
    img_compressed = (np.dstack((red_inverted, green_inverted, blue_inverted))).astype(np.uint8)
    return img_compressed


def compress_image(img_filename, n_components=20):
    print(f"Compress '{img_filename}' using PCA({n_components})...", end="")
    img = load_image(img_filename)
    r, g, b = split_image_RGB(img)
    ri, gi, bi = apply_pca_RGB(r, g, b, n_components=n_components)
    img_compressed = compress(ri, gi, bi)
    plt.imshow(img_compressed)      # view the compressed image
    print("Done")
    return img_compressed


def run_principal_component_analysis(img_filename):
    df, digits = import_dataset()
    view_data(df)
    converted_data = reduce_image_dimensions(df)
    visualize_results(converted_data, digits)

    # Image compression using PCA
    compress_image(img_filename, n_components=20)
    compress_image(img_filename, n_components=50)
    compress_image(img_filename, n_components=100)
    compress_image(img_filename, n_components=200)


if __name__ == "__main__":
    img_dir = '/Users/michael/git/nashed/t1/webTraderApp/files/in'
    img_filename = os.path.join(img_dir, 'my_doggo_sample.jpg')
    run_principal_component_analysis(img_filename)


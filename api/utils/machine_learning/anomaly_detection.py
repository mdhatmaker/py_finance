import pycaret
from pycaret.datasets import get_data
from pycaret.anomaly import *

# https://www.askpython.com/python-modules/detecting-anomalies-pycaret
# https://pycaret.gitbook.io/docs/get-started/quickstart#anomaly-detection


# Loading and splitting the data
dataset = get_data('mice')

train = dataset.sample(frac = 0.95,random_state = 786)
train.head()

test = dataset.drop(train.index)
test.head()

train.reset_index(drop=True,inplace=True)
test.reset_index(drop=True,inplace=True)


# Set up anomaly detection
anomaly_setup = setup(train,normalize = True,session_id = 123)

# The models function is used to print all the available models from the anomaly module.
models()

# Use the Isolation Forest for our use case
iforest = create_model("iforest")
print(iforest)

# Create and evaluate the model
result = assign_model(iforest)
result.head()

# 3d visualization of the anomalies in the train data
plot_model(iforest)

# 2d visualization
plot_model(iforest, plot="umap")

# Make predictions using the model on the test dataset
predictions = predict_model(iforest,test)
predictions.head()

# Save the model
save_model(iforest,"iforestmodel")


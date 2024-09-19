from pycaret.classification import *
from pycaret.datasets import get_data
from pycaret.regression import *

# https://www.askpython.com/python-modules/pycaret
# https://pycaret.gitbook.io/docs/


# clf1 = setup(data=diabetes, target='Class variable', session_id=123)
# create_model('lr')

data = get_data('insurance')

# Now, it is time to set up the initial model pipeline. Since we are looking at a regression example,
# the prediction label is obviously charges. Hence, we pass this as a target to the setup function.
s = setup(data, target='charges')

best = compare_models()
print(best)

gbr = create_model('gbr')   # GradientBoostingRegressor

evaluate_model(gbr)

tuned_gbr = tune_model(gbr)

# can also use gbr instead of best
plot_model(best)

final_best = finalize_model(best)
print(final_best)

# The create_app function uses gradio to create a demo application in the notebook itself for inference
# using the features of the data. However, if you are interested, the same can be deployed in Streamlit.
create_app(final_best)

# Save the model in pickle format
save_model(final_best,'my_gbr')


"""
Pycaret is a low-code machine learning library that automates the ML workflow, making the process
seamless and productive. It can reduce hundreds of lines of code to just a few, and can be easily
integrated with BI platforms like Power BI and Tableau for creating interactive dashboards.
"""

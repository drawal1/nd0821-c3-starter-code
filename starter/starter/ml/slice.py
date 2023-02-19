"""
    author: Dhar Rawal
    function that returns slice performance
"""

from starter.starter.ml.data import process_data
import starter.starter.ml.model as model

def slice_performance(data, cat_features, trained_model, encoder, lb, slicing_cat):
    """ returns dictionary of slice performance for slicing_cat
        data is the full set of data
        cat_features is a list of cat feature names
        model is a trained model
    """
    slice_performance_dict = {}
    for cat_value in data[slicing_cat].unique():
        # extract the slice
        sliced_data = data[data[slicing_cat] == cat_value]

        x_inf, y_inf, _, _ = process_data(
            sliced_data, categorical_features=cat_features, label="salary", training=False,
            encoder=encoder, lb=lb
        )

        y_pred = model.inference(trained_model, x_inf)

        precision, recall, fbeta = model.compute_model_metrics(y_inf, y_pred)

        slice_performance_dict[cat_value] = (precision, recall, fbeta)

    return slice_performance_dict

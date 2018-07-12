# Sora
Visualization toolkit for ML and DL methods

------------
## Version 1.0.0

### New features:
- Support calculating SHAP values for scikit-learn *decision tree regressor*.
- Sample script to show how the result would look like.

### How to calculate it?
1. Clone or download this repository and go into the root of the toolkit.
2. In your code, do as follows:
    ```python
    from SHAP.tree import TreeExplainer

    # model is your own sklearn.tree.DecisionTreeRegressor()
    # Predictors: labels of x
    explainer = TreeExplainer(model).shap_values(x=data[0])
    print(explainer[0, :])

    plt.bar(range(len(predictors)), explainer[0, :-1], tick_label=predictors)
    ```

### How to run the sample script:
1. Clone or download this repository and go into the root of the toolkit.
2. Run ```$ python sample_script.tree.decision_tree_sample.py```

Then you are likely to see a ```shap.png``` in the pwd. The picture is a bar figure and should look like 
![](http://imglf3.nosdn0.126.net/img/UFZ3T1ZWbXcvWlRvLzd5elliNUpTQ1dBZlcyeFNZcy9ubjF2NlZ5cVZRQzkyUmdxUWErZEdRPT0.png?imageView&thumbnail=500x0&quality=96&stripmeta=0)

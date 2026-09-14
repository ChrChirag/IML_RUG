import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    """
    To ideally determine which features to use, we must group each of the columns
    and calculate their means and standard deviations.
    If the means of the groups are similar and the standard deviation is small then the groups barely overlap.
    """

    dataset = load_wine(
        as_frame=True
    )  # true changes the output from a numpy array to a pandas dataframe
    print(type(dataset))
    # print(dataset.keys())

    # pandas dataframe and feature names
    # print(dataset.feature_names)
    # print(dataset.target_names)
    # print(dataset.target)
    # print(dataset.frame)
    # print(dataset.data)
    # print(dataset.DESCR)

    # we put out the wine data in X and the targets in y
    X = dataset.data
    y = dataset.target
    print(X.shape)  # (instances, attributes)
    print(y.shape)

    print(type(dataset.frame))
    print(
        dataset.frame.groupby("target").mean()
    )  # compare the means of the groups to see if there is a signiicance
    print(
        dataset.frame.groupby("target").std()
    )  # using the mean leverage the stds to see if there is a signiicance between the groups
    means = dataset.frame.groupby("target").mean()
    stds = dataset.frame.groupby("target").std()

    mean_diff_classoneandtwo = (
        means.loc[1] - means.loc[2]
    )  # finding mean differences between classes 1 and 2 to test for significance
    avg_std_classoneandtwo = (
        (stds.loc[1] + stds.loc[2]) / 2
    )  # finding std avg between two classes (1 and 2) so it can be used to find the amount of deviations
    significant_feature_classoneandtwo = (
        mean_diff_classoneandtwo / avg_std_classoneandtwo
    )  # OD280 (2.87), color intensity (2.66), flavanoids (2.60) and hue (2.35) are most significant

    mean_diff_classzeroandtwo = (
        means.loc[0] - means.loc[2]
    )  # finding mean differences between classes 0 and 2 to test for significance
    avg_std_classzeroandtwo = (
        (stds.loc[0] + stds.loc[2]) / 2
    )  # finding std avg between two classes (0 and 2) so it can be used to find the amount of deviations
    significant_feature_classzeroandtwo = (
        mean_diff_classzeroandtwo / avg_std_classzeroandtwo
    )  # flavanoids (6.37), OD280 (4.69) and hue (3.29)

    mean_diff_classzeroandone = (
        means.loc[0] - means.loc[1]
    )  # finding mean differences between classes 0 and 1 to test for significance
    avg_std_classzeroandone = (
        (stds.loc[0] + stds.loc[1]) / 2
    )  # finding std avg between two classes (0 and 1) so it can be used to find the amount of deviations
    significant_feature_classzeroandone = (
        mean_diff_classzeroandone / avg_std_classzeroandone
    )  # proline (3.15), alcohol (2.93), and (2.26)

    print(significant_feature_classoneandtwo)
    print(significant_feature_classzeroandtwo)
    print(significant_feature_classzeroandone)

    # ideal feature picks: flavanoids, proline, hue, OD280 and alcohol
    feature_picks = [
        "flavanoids",
        "proline",
        "hue",
        "od280/od315_of_diluted_wines",
        "alcohol",
    ]
    feature_table = dataset.frame[feature_picks]
    print(feature_table)

    # log reg model with the selected features
    # we split the data into training and testing sets in a shuffle and then stratisfied format.
    X_train, X_test, y_train, y_test = train_test_split(
        feature_table, y, test_size=0.2, random_state=42, stratify=y
    )
    # realized that the features dont converge even at max iterations, so using stadardization will solve this.
    wine_scaler = StandardScaler()
    X_train = wine_scaler.fit_transform(X_train)
    X_test = wine_scaler.transform(X_test)
    print(X_train.shape, X_test.shape)  # to verify the shapes

    # log reg model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
    # the feautures are linearly separable because the wine groups are so distinct that they sit in separae piles meaning logistic regression can easily grpah through these points.

    # The assignment now asks to compare your results with training the same model on the full set of features.

    #TO DO SO:
    #create a second split with 13 features instead of 5
    #then split the training set and test set
    #then train and test the model

    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        dataset.data, y, test_size=0.2, random_state=42, stratify=y
    )
   
    wine_scaler = StandardScaler()
    X_train_full = wine_scaler.fit_transform(X_train_full)
    X_test_full = wine_scaler.transform(X_test_full)
    print(X_train_full.shape, X_test_full.shape)  # to verify the shapes

    model_full = LogisticRegression()
    model_full.fit(X_train_full, y_train_full)
    print(model_full.score(X_test_full, y_test_full)) 
    #score of 0.977 which is barely any different from the 5 feature model

    
    #The honest conclusion is that both models perform about equally well, around 97 to 100%, 
    #and that a single split can't distinguish them. 
    #It's also a more interesting result than it first looks as five features do as well as all 13. 
    #That's important, since it means the eight features dropped didn't improve the predictions once these five were in.
    
#feature tranformation and residual
    #feature transformation using log, square, root and exponential

    
    X_transformed = pd.DataFrame()
    X_transformed["flavanoids_sq"] = feature_table["flavanoids"] ** 2
    X_transformed["proline_log"] = np.log1p(feature_table["proline"]) #using log1p to avoid working with zeros and negative values
    X_transformed["hue_cube"] = feature_table["hue"] ** 3
    X_transformed["od280_sqrt"] = np.sqrt(feature_table["od280/od315_of_diluted_wines"])
    X_transformed["alcohol_exp"] = np.exp(feature_table["alcohol"] / 10)

    X_train_transformed, X_test_transformed, y_train_transformed, y_test_transformed = train_test_split(
            X_transformed, y, test_size=0.2, random_state=42, stratify=y)

    transformed_scaler = StandardScaler()
    X_train_transformed_scaled = transformed_scaler.fit_transform(X_train_transformed)
    X_test_transformed_scaled = transformed_scaler.transform(X_test_transformed)

    #fitting the new model with the transformed features
    model_transformed = LogisticRegression(random_state=42)
    model_transformed.fit(X_train_transformed_scaled, y_train_transformed)

    #testing the model
    test_score = model_transformed.score(X_test_transformed_scaled, y_test_transformed)
    print(test_score)

    #residuals (difference between actual and prediction)
    test_probabilities = model_transformed.predict_proba(X_test_transformed_scaled)
    actual_class_test_proba = test_probabilities[
        np.arange(len(y_test_transformed)), y_test_transformed
    ]
    test_residuals = 1.0 - actual_class_test_proba
    print(test_residuals)

    #plotting the residuals

    plt.figure(figsize=(10, 6))
    plt.axhline(
        y=0, color="black", linestyle="--", alpha=0.5, label="Perfect Confidence Floor"
    )

    scatter = plt.scatter(
        X_test_transformed["proline_log"],
        test_residuals,
        c=y_test_transformed,
        cmap="viridis",
        edgecolors="k",
        alpha=0.8,
        s=60,
    )

    plt.xlabel("Log-Transformed Proline Content")
    plt.ylabel("Residual Error (1.0 - Predicted Probability)")
    plt.title("Residual Plot: Testing Generalizable Linear Separability")

    handles, _ = scatter.legend_elements()
    plt.legend(handles, dataset.target_names, title="Wine Classes")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("residual_plot.png")
"""
The model is very confident at the extremes but less certain in the middle 
because the various proline values isolate distinct classes at far ends, 
but then overlap significantly in the center.
"""

if __name__ == "__main__":
    main()

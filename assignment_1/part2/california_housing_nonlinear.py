"""
Assignment 1 - Part IIc: California housing (nonlinear)

Code cells extracted from california_housing_nonlinearity.ipynb, in notebook order.
"""

# %% cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score, mean_squared_error

# %matplotlib inline  # notebook-only magic, commented out for the .py version
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.float_format", "{:.3f}".format)

RANDOM_STATE = 42
print("scikit-learn", sklearn.__version__, "| pandas", pd.__version__, "| numpy", np.__version__)

# %% cell 2
housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target
df = housing.frame

print(f"{X.shape[0]:,} districts, {X.shape[1]} features, target = {housing.target_names[0]}")
df.head()

# %% cell 3
df.describe().T

# %% cell 4
n_capped = (y >= 5.0).sum()
print(f"Median AveOccup: {X['AveOccup'].median():.2f}   maximum: {X['AveOccup'].max():,.0f}")
print(f"Median AveRooms: {X['AveRooms'].median():.2f}   maximum: {X['AveRooms'].max():,.0f}")
print(f"Districts at the 5.00001 cap: {n_capped} ({n_capped / len(y):.1%})")
print(f"Districts at HouseAge = 52 (also a top code): {(X['HouseAge'] == 52).sum()}")

# %% cell 5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print(f"Training set: {len(X_train):,} districts | test set: {len(X_test):,} districts")

# %% cell 6
relevance = pd.DataFrame({
    "Pearson r": X_train.corrwith(y_train),
    "Spearman rho": X_train.corrwith(y_train, method="spearman"),
    "Mutual information": mutual_info_regression(X_train, y_train, random_state=RANDOM_STATE),
}).sort_values("Mutual information", ascending=False)

relevance

# %% cell 7
order = relevance.index[::-1]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
panels = [("Pearson r", "|Pearson r|  (linear)"),
          ("Spearman rho", "|Spearman rho|  (monotonic)"),
          ("Mutual information", "Mutual information  (any shape)")]
for ax, (col, title) in zip(axes, panels):
    ax.barh(order, relevance.loc[order, col].abs(), color="tab:blue")
    ax.set_title(title, fontsize=10)
fig.suptitle("How strongly each feature is related to house value (training set)", y=1.02)
plt.tight_layout()
plt.show()

# %% cell 8
corr = X_train.corr()
upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
pairs = upper.stack().rename("r").reset_index()
pairs.columns = ["feature 1", "feature 2", "r"]
pairs[pairs["r"].abs() > 0.5].sort_values("r", key=np.abs, ascending=False)

# %% cell 9
dropped = ["AveBedrms", "Population"]
all_features = list(X.columns)
selected = [f for f in all_features if f not in dropped]

print("Selected:", selected)
print("Dropped: ", dropped)

# %% cell 10
PARAM_GRID = {"max_depth": [4, 6, 8, 10, 12, 14, 16, None],
              "min_samples_leaf": [1, 5, 10, 20, 40, 80]}


def tuned_tree(X_tr, y_tr):
    # Choose max_depth and min_samples_leaf by 5-fold CV on the training data only.
    search = GridSearchCV(DecisionTreeRegressor(random_state=RANDOM_STATE), PARAM_GRID,
                          cv=cv, scoring="r2", n_jobs=-1)
    search.fit(X_tr, y_tr)
    return search.best_estimator_, search.best_params_


def compare_models(features, X_tr=None, X_te=None):
    # Fit linear regression and a tuned tree on the same features and score both the same way.
    X_tr = X_train if X_tr is None else X_tr
    X_te = X_test if X_te is None else X_te
    tree, best_params = tuned_tree(X_tr[features], y_train)
    models = {"Linear regression": LinearRegression(), "Decision tree": tree}
    rows, folds = {}, {}
    for name, model in models.items():
        fold_r2 = cross_val_score(model, X_tr[features], y_train, cv=cv, scoring="r2")
        model.fit(X_tr[features], y_train)
        pred = model.predict(X_te[features])
        folds[name] = fold_r2
        rows[name] = {
            "CV R² mean": fold_r2.mean(),
            "CV R² sd": fold_r2.std(),
            "Train R²": r2_score(y_train, model.predict(X_tr[features])),
            "Test R²": r2_score(y_test, pred),
            "Test RMSE (USD)": np.sqrt(mean_squared_error(y_test, pred)) * 100_000,
        }
    table = pd.DataFrame(rows).T
    table["Test RMSE (USD)"] = table["Test RMSE (USD)"].round(-2).astype(int)
    folds = pd.DataFrame(folds, index=[f"fold {i}" for i in range(1, 6)])
    folds["Tree minus linear"] = folds["Decision tree"] - folds["Linear regression"]
    return {"table": table, "folds": folds, "models": models, "best_params": best_params}


def one_feature_check(feature_sets, X_tr):
    # Linear regression vs a tree on one feature (or a small group) at a time, 5-fold CV R².
    rows = {}
    for label, cols in feature_sets.items():
        lin = cross_val_score(LinearRegression(), X_tr[cols], y_train, cv=cv, scoring="r2").mean()
        search = GridSearchCV(DecisionTreeRegressor(random_state=RANDOM_STATE),
                              {"min_samples_leaf": [10, 25, 50, 100, 200, 400]}, cv=cv, scoring="r2")
        tree = search.fit(X_tr[cols], y_train).best_score_
        rows[label] = {"Linear CV R²": lin, "Tree CV R²": tree, "Gap": tree - lin}
    return pd.DataFrame(rows).T

# %% cell 11
unpruned = DecisionTreeRegressor(random_state=RANDOM_STATE)
unpruned_cv = cross_val_score(unpruned, X_train[selected], y_train, cv=cv, scoring="r2").mean()
unpruned.fit(X_train[selected], y_train)
print(f"Unpruned tree: train R² = {unpruned.score(X_train[selected], y_train):.3f}, CV R² = {unpruned_cv:.3f}, "
      f"{unpruned.get_n_leaves():,} leaves")

# %% cell 12
sel = compare_models(selected)
print("Tree hyperparameters chosen by CV:", sel["best_params"])
sel["table"]

# %% cell 13
sel["folds"]

# %% cell 14
lin_pred = sel["models"]["Linear regression"].predict(X_test[selected])
tree_pred = sel["models"]["Decision tree"].predict(X_test[selected])
print(f"Linear regression: {(lin_pred > 5.00001).sum()} test predictions above the 5.00001 cap, "
      f"{(lin_pred < 0).sum()} below zero (range {lin_pred.min():.2f} to {lin_pred.max():.2f})")
print(f"Decision tree:     predictions range from {tree_pred.min():.2f} to {tree_pred.max():.2f}")

# %% cell 15
feature_sets = {f: [f] for f in selected}
feature_sets["Latitude + Longitude"] = ["Latitude", "Longitude"]
per_feature = one_feature_check(feature_sets, X_train)
per_feature

# %% cell 16
def binned_means(x, y, bins=30):
    groups = pd.qcut(x, q=bins, duplicates="drop")
    means = pd.DataFrame({"x": x, "y": y}).groupby(groups, observed=True).mean()
    return means["x"], means["y"]


def shape_plot(ax, x, y, title, sample=3000):
    # Grey dots: a random sample of districts. Orange: mean house value in 30 equal-count bins.
    # Blue: least-squares line fitted to all training districts. x-axis shows the 1st-99th percentile.
    rng = np.random.default_rng(0)
    idx = rng.choice(len(x), size=min(sample, len(x)), replace=False)
    ax.scatter(x.iloc[idx], y.iloc[idx], s=3, alpha=0.15, color="grey")
    bx, by = binned_means(x, y)
    ax.plot(bx, by, "o-", color="tab:orange", ms=3, lw=1.5, label="binned mean")
    line = LinearRegression().fit(x.to_frame(), y)
    lo, hi = x.quantile(0.01), x.quantile(0.99)
    grid = pd.DataFrame({x.name: np.linspace(lo, hi, 100)})
    ax.plot(grid[x.name], line.predict(grid), color="tab:blue", lw=2, label="linear fit")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 5.3)
    ax.set_title(title, fontsize=10)


fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, f in zip(axes.ravel(), selected):
    shape_plot(ax, X_train[f], y_train, f)
axes[0, 0].legend(loc="upper left", fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel("MedHouseVal (100k USD)")
fig.suptitle("Shape of each relationship: binned means vs the best straight line (training set)", y=1.0)
plt.tight_layout()
plt.show()

# %% cell 17
coords = X_test[["Latitude", "Longitude"]].to_numpy()
neighbours = NearestNeighbors(n_neighbors=11).fit(coords).kneighbors(coords, return_distance=False)[:, 1:]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharex=True, sharey=True, constrained_layout=True)
for ax, name in zip(axes, ["Linear regression", "Decision tree"]):
    resid = (y_test - sel["models"][name].predict(X_test[selected])).to_numpy()
    spatial_r = np.corrcoef(resid, resid[neighbours].mean(axis=1))[0, 1]
    order = np.argsort(np.abs(resid))
    sc = ax.scatter(X_test["Longitude"].to_numpy()[order], X_test["Latitude"].to_numpy()[order],
                    c=resid[order], cmap="RdBu", vmin=-2, vmax=2, s=5)
    ax.set_title(f"{name}\ncorrelation with neighbours' residuals = {spatial_r:.2f}", fontsize=10)
    ax.set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
fig.colorbar(sc, ax=axes, label="actual minus predicted (100k USD)", shrink=0.9)
fig.suptitle("Test-set residuals on the map (blue = predicted too low, red = predicted too high)")
plt.show()

# %% cell 18
full = compare_models(all_features)
print("Tree hyperparameters chosen by CV:", full["best_params"])
full["table"]

# %% cell 19
comparison = pd.concat({"6 selected features": sel["table"], "All 8 features": full["table"]})
comparison

# %% cell 20
imp = permutation_importance(full["models"]["Decision tree"], X_test[all_features], y_test,
                             n_repeats=10, random_state=RANDOM_STATE, scoring="r2")
importance = pd.DataFrame({"Drop in test R² when shuffled": imp.importances_mean,
                           "sd over 10 shuffles": imp.importances_std},
                          index=all_features).sort_values("Drop in test R² when shuffled", ascending=False)
importance

# %% cell 21
to_log = ["AveRooms", "AveOccup"]
X_log = X.copy()
X_log[to_log] = np.log(X_log[to_log])
X_log_train, X_log_test = X_log.loc[X_train.index], X_log.loc[X_test.index]

pd.DataFrame({"Skewness before": X_train[to_log].skew(),
              "Skewness after log": X_log_train[to_log].skew()})

# %% cell 22
positive = ["MedInc", "HouseAge", "AveRooms", "AveOccup"]
raw = one_feature_check({f: [f] for f in positive}, X_train)
logged = one_feature_check({f: [f] for f in positive}, np.log(X_train[positive]))

pd.DataFrame({"Linear CV R², x": raw["Linear CV R²"],
              "Linear CV R², ln x": logged["Linear CV R²"],
              "Tree CV R², x": raw["Tree CV R²"],
              "Tree CV R², ln x": logged["Tree CV R²"]})

# %% cell 23
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for col, f in enumerate(to_log):
    shape_plot(axes[0, col], X_train[f], y_train, f)
    shape_plot(axes[1, col], X_log_train[f].rename(f"ln {f}"), y_train, f"ln({f})")
axes[0, 0].legend(loc="upper left", fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel("MedHouseVal (100k USD)")
fig.suptitle("AveRooms and AveOccup before (top) and after (bottom) the log", y=1.0)
plt.tight_layout()
plt.show()

# %% cell 24
logm = compare_models(selected, X_log_train, X_log_test)
print("Tree hyperparameters chosen by CV:", logm["best_params"])
logm["table"]

# %% cell 25
logm["folds"]

# %% cell 26
identical = np.array_equal(sel["models"]["Decision tree"].predict(X_test[selected]),
                           logm["models"]["Decision tree"].predict(X_log_test[selected]))
print("Tree test predictions identical before and after the log:", identical)

gap_before = sel["table"].loc["Decision tree", "CV R² mean"] - sel["table"].loc["Linear regression", "CV R² mean"]
gap_after = logm["table"].loc["Decision tree", "CV R² mean"] - logm["table"].loc["Linear regression", "CV R² mean"]
print(f"CV gap (tree minus linear): {gap_before:.3f} before the log, {gap_after:.3f} after "
      f"-> {1 - gap_after / gap_before:.0%} of the gap removed")

# %% cell 27
runs = {"6 selected": sel, "All 8": full, "6 selected, log(AveRooms), log(AveOccup)": logm}
summary = pd.DataFrame({
    label: {
        "Linear CV R²": r["table"].loc["Linear regression", "CV R² mean"],
        "Tree CV R²": r["table"].loc["Decision tree", "CV R² mean"],
        "Gap (CV)": r["table"].loc["Decision tree", "CV R² mean"] - r["table"].loc["Linear regression", "CV R² mean"],
        "Linear test R²": r["table"].loc["Linear regression", "Test R²"],
        "Tree test R²": r["table"].loc["Decision tree", "Test R²"],
        "Gap (test)": r["table"].loc["Decision tree", "Test R²"] - r["table"].loc["Linear regression", "Test R²"],
    }
    for label, r in runs.items()
}).T
summary

# %% cell 28
labels = ["6 selected", "All 8", "6 selected,\nlog(AveRooms), log(AveOccup)"]
pos = np.arange(len(runs))
width = 0.36
fig, ax = plt.subplots(figsize=(9, 4.2))
for k, (name, colour) in enumerate([("Linear regression", "tab:blue"), ("Decision tree", "tab:orange")]):
    means = [r["table"].loc[name, "CV R² mean"] for r in runs.values()]
    sds = [r["table"].loc[name, "CV R² sd"] for r in runs.values()]
    bars = ax.bar(pos + (k - 0.5) * width, means, width, yerr=sds, capsize=4, color=colour, label=name)
    ax.bar_label(bars, labels=[f"{m:.3f}" for m in means], padding=6, fontsize=9)
ax.set_xticks(pos)
ax.set_xticklabels(labels)
ax.set_ylim(0, 0.95)
ax.set_ylabel("5-fold CV R² (mean ± sd)")
ax.set_title("Linear regression vs decision tree on each feature set")
ax.legend(loc="upper left", ncol=2, frameon=False)
plt.tight_layout()
plt.show()

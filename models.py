
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR

def get_models():
    return {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=15, random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=50, max_depth=15,
            n_jobs=4, random_state=42
        ),

        "KNN": KNeighborsRegressor(
            n_neighbors=5, n_jobs=4
        ),

        "SVR": LinearSVR(
            max_iter=3000, random_state=42
        ),

        "Ridge Regression": Ridge(alpha=10),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=50, max_depth=15,
            n_jobs=4, random_state=42
        ),

        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=100, max_depth=10,
            random_state=42
        )
    }

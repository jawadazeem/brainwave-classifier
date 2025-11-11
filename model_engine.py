from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def train_and_evaluate(processed_data):
    X = processed_data.drop(columns=["Condition"])
    y = processed_data["Condition"]

    if len(processed_data) < 2:
        raise ValueError("Need at least two feature rows to train and evaluate the model")

    stratify = y if y.value_counts().min() >= 2 else None
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=100, random_state=42),
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=stratify,
    )
    
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

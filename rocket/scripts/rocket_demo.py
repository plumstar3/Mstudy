from sktime.datasets import load_arrow_head
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sktime.classification.kernel_based import RocketClassifier


# sktimeの組み込みデータセット（1変量時系列データ）を読み込み
X, y = load_arrow_head(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --- RocketClassifier (rocket_transform="rocket" が ROCKET 本体) ---
rocket_clf = RocketClassifier(rocket_transform="rocket", num_kernels=10000, random_state=42)
rocket_clf.fit(X_train, y_train)

y_pred = rocket_clf.predict(X_test)
print(f"ROCKET Accuracy: {accuracy_score(y_test, y_pred):.3f}")
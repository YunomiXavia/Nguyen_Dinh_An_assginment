from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

def train_model(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    C_values = np.logspace(-3, 3, 10)
    best_accuracy = 0
    best_C = 0
    best_clf = None

    for C in C_values:
        clf = svm.SVC(kernel='linear', C=C, probability=True)
        clf.fit(X_train, y_train)

        y_val_pred = clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)
        print(f"Độ chính xác với C={C}: {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_C = C
            best_clf = clf

    print(f"Giá trị C tốt nhất: {best_C} với độ chính xác: {best_accuracy} (độ chính xác càng lớn càng tốt)")

    W = best_clf.coef_[0]
    b = best_clf.intercept_[0]

    return scaler, best_clf, W, b, X_train, y_train, X_val, y_val

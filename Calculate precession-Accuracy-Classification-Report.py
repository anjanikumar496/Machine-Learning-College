

### Calculate precession, Accuracy , Classification Report

from import_package import  DataPreprocessin

imported_data = DataPreprocessin.Process()

from  sklearn.svm import SVC
svc = SVC()

svc.fit(imported_data.get('X_train'),imported_data.get('y_train'))
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = svc.predict(imported_data.get('X_test'))
print('Confusion Matrix : \n', confusion_matrix(imported_data.get('y_test'), imported_data.get('y_pred')))
print('Classification Report : \n', classification_report(imported_data.get('y_test'), imported_data.get('y_pred')))
print('Accuracy Metrics : \n', accuracy_score(imported_data.get('y_test'), imported_data.get('y_pred')))
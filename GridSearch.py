

### Calculate precession, Accuracy , Classification Report

from data_preprocessing import  DataPreprocessin

imported_data = DataPreprocessin.Process()

from  sklearn.svm import SVC
svc = SVC()

svc.fit(imported_data.get('X_train'),imported_data.get('y_train'))

from sklearn.model_selection import GridSearchCV

param_grid = {'C':[0.1,1,10,100,1000],'gamma':[0.0001,0.001,0.1,0.01],'kernel':['rbf']}
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


### Create a GridSearchCV object and fit it to the training data.

cv = GridSearchCV(SVC(),param_grid=param_grid,refit=True,verbose=4,return_train_score=True)
cv.fit(imported_data.get('X_train'),imported_data.get('y_train'))

print(cv.best_params_)
print(cv.best_estimator_)

y_pred = cv.predict(imported_data.get('X_train'))
print('Confusion Matrix : \n', confusion_matrix(imported_data.get('y_train'), imported_data.get('y_pred')))
print('Classification Report : \n', classification_report(imported_data.get('y_train'), imported_data.get('y_pred')))
print('Accuracy Metrics : \n', accuracy_score(imported_data.get('y_train'), imported_data.get('y_pred')))


y_pred = cv.predict(imported_data.get('X_test'))
print('Confusion Matrix : \n', confusion_matrix(imported_data.get('y_test'),imported_data.get('y_pred') ))
print('Classification Report : \n', classification_report(imported_data.get('y_test'), imported_data.get('y_pred')))
print('Accuracy Metrics : \n', accuracy_score(imported_data.get('y_test'), imported_data.get('y_pred')))

### You should have done about the same or exactly the same, this makes sense, there is basically just one point that is too noisey to grab, which makes sense, we don't want to have an overfit model that would be able to grab that.

### Great Job!!!



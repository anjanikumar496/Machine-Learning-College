from data_preprocessing import  DataPreprocessin

imported_data = DataPreprocessin.Process()

from  sklearn.svm import SVC
svc = SVC()

print(svc.fit(imported_data.get('X_train'),imported_data.get('y_train')))


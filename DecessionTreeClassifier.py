
from data_preprocessing import  DataPreprocessin

imported_data = DataPreprocessin.Process()

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(imported_data.get('X_train'),imported_data.get('y_train'))


predictions = classifier.predict(imported_data.get('X_test'))
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(imported_data.get('y_test'),predictions))


confusion_matrix(imported_data.get('y_test'),predictions)



# So the Above metrix is indicating that 26 is the kind of observation which achived the Target and Machine also identified the same, 23 are the those observation that not achived the Target and Machine also Identified the same
# 3 are those who achived the Target but machine Identified not Achived
# 9 are those who not achived the Target but Machine acived yes it achived
# so Decission Tree is givin the 80%


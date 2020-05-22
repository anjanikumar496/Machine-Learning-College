
from DecessionTreeClassifier import  *

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(imported_data.get('X_train'),imported_data.get('y_train'))

predctions=rfc.predict(imported_data.get('X_test'))

print(classification_report(imported_data.get('y_test'),predctions))

confusion_matrix(imported_data.get('y_test'),predctions)

#
# So the Above metrix is indicating that 24 is the kind of observation which achived the Target and Machine also identified the same, 27 are the those observation that not achived the Target and Machine also Identified the same
# 5 are those who achived the Target but machine Identified not Achived
# 5 are those who not achived the Target but Machine acived yes it achived
# so RandomForest is giving the 84%
# So I could say Random Forest giving the much better accuracy



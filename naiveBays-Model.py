

from DecessionTreeClassifier import  *

from sklearn.naive_bayes import MultinomialNB

nbclf = MultinomialNB()
nbclf.fit(imported_data.get('X_train'),imported_data.get('y_train'))


predictions = nbclf.predict(imported_data.get('X_test'))

print(classification_report(predictions,imported_data.get('y_test')))

print(confusion_matrix(predictions,imported_data.get('y_test')))


# So the Above metrix is indicating that 26 is the kind of observation which achived the Target and Machine also identified the same, 26 are the those observation that not achived the Target and Machine also Identified the same
# 6 are those who achived the Target but machine Identified not Achived
# 3 are those who not achived the Target but Machine acived yes it achived
# so NaiveBays is giving the 85%
# So I could say Naive Bays giving the much better accuracy



### Error GridSearch.py  line 29-31   and   35-37

#### Error Calculate precession-Accuracy-Classification-Report.py    line 16-18
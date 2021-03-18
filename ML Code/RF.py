from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.ensemble import RandomForestRegressor

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import pickle 
# load the dataset
data = open('ManualAnnotatedFakeNewsDataset.txt').read()
#data = open('AutomaticAnnotatedFakeNewsDataset.txt').read()
labels, texts = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split("\t")
    labels.append(content[0])
    texts.append(" ".join(content[1:]))

# create a dataframe using texts and lables
trainDF = pandas.DataFrame()
trainDF['tweet'] = texts
trainDF['class'] = labels

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['tweet'], trainDF['class'])




# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['tweet'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['tweet'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['tweet'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['tweet'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 


# RandomForestRegressor(n_estimators=20, random_state=0)

def train_model(classifier, feature_vector_train, label, feature_vector_valid, modelName, is_neural_net=False):
    # fit the training dataset on the classifier
    clf = classifier.fit(feature_vector_train, label)
    with open(modelName, 'wb') as picklefile:
        pickle.dump(clf,picklefile)
    # predict the labels on validation dataset
    predictions = clf.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)


# RF on Count Vectors
RFmodelname = "40CountVectors_RF_Model"
accuracy = train_model(RandomForestRegressor(n_estimators=20, random_state=0), xtrain_count, train_y, xvalid_count,RFmodelname)
print ("RF, Count Vectors: ", accuracy)

# RF on Word Level TF IDF Vectors
RFmodelname = "41WordLevel_RF_Model"
accuracy = train_model(RandomForestRegressor(n_estimators=20, random_state=0), xtrain_tfidf, train_y, xvalid_tfidf,RFmodelname)
print ("RF, WordLevel TF-IDF: ", accuracy)



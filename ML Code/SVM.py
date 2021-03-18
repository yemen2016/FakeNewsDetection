from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
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
#stemming
data1 = []
from nltk import word_tokenize

from nltk.stem.isri import ISRIStemmer

st = ISRIStemmer()
for tx in texts:
    tweet = ""
    for a in word_tokenize(tx):
        tweet = tweet + st.stem(a)+ " "
    data1.append(tweet.strip())

#print(data1[:10])
#tashfeen
data2 = []
import pyarabic.arabrepr
arepr = pyarabic.arabrepr.ArabicRepr()
repr = arepr.repr
from tashaphyne.stemming import ArabicLightStemmer
ArListem = ArabicLightStemmer()
for tx in texts:
    tweet = ""
    for a in word_tokenize(tx):
        stem = ArListem.light_stem(a)
        tweet = tweet + ArListem.get_stem()+ " "
        #tweet = tweet + ArListem.get_root()+ " "
    data2.append(tweet.strip())
#print(data2[:10])



# create a dataframe using texts and lables
trainDF = pandas.DataFrame()
trainDF['tweet'] = data2
trainDF['class'] = labels

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['tweet'], trainDF['class'],test_size = 0.2)




from nltk.corpus import stopwords
stopwords_list = stopwords.words('arabic')

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['tweet'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

# word level tf-idf
#, stop_words=set(stop)
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=30000)
tfidf_vect.fit(trainDF['tweet'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,3), max_features=30000)
tfidf_vect_ngram.fit(trainDF['tweet'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=30000)
tfidf_vect_ngram_chars.fit(trainDF['tweet'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 

def train_model(classifier, feature_vector_train, label, feature_vector_valid, modelName, is_neural_net=False):
    # fit the training dataset on the classifier
    clf = classifier.fit(feature_vector_train, label)
    with open(modelName, 'wb') as picklefile:
        pickle.dump(clf,picklefile)
    # predict the labels on validation dataset
    predictions = clf.predict(feature_vector_valid)
    #scores = cross_validate(clf, feature_vector_train, label, cv=10, scoring='f1_weighted')
    #print(scores)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    f = open('FakeNews/results.txt', 'a+')
    print(metrics.precision_score(predictions, valid_y, average='weighted'))
    f.write(str(metrics.precision_score(predictions, valid_y, average='weighted'))+"\t")
    print(metrics.recall_score(predictions, valid_y, average='weighted'))
    f.write(str(metrics.recall_score(predictions, valid_y, average='weighted'))+"\t")
    f.write(str(metrics.f1_score(predictions, valid_y, average='weighted'))+"\n")
    f.close()
    return metrics.f1_score(predictions, valid_y, average='weighted')



SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVMmodelname = "FakeNews/30CountVectors_SVM_Model"
# SVM Classifier on Count Vectors
clf = svm.SVC(kernel='linear')
accuracy = train_model(clf, xtrain_count, train_y, xvalid_count,SVMmodelname)
print ("SVM, Count Vectors: ", accuracy)

SVMmodelname = "FakeNews/31WordLevel_SVM_Model"
# SVM Classifier on Word Level TF IDF Vectors
accuracy = train_model(clf, xtrain_tfidf, train_y, xvalid_tfidf,SVMmodelname)
print ("SVM, WordLevel TF-IDF: ", accuracy)

SVMmodelname = "FakeNews/32N-GramVectors_SVM_Model"
# SVM Classifier on Ngram Level TF IDF Vectors
accuracy = train_model(clf, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram,SVMmodelname)
print ("SVM, N-Gram Vectors: ", accuracy)

SVMmodelname = "FakeNews/33CharLevelVectors_SVM_Model"
# SVM Classifier on Character Level TF IDF Vectors
accuracy = train_model(clf, xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars,SVMmodelname)
print ("SVM, CharLevel Vectors: ", accuracy)


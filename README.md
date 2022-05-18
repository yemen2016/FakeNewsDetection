# FakeNewsDetection

<b>Arabic Fake News corpora: </b></br>
The following are two Arabic corpora for the task of fake news detection:
1.	Manual Annotated Corpus: 

The annotation process resulted in a corpus containing 1,537 tweets (835 fake and 702 genuine), after excluding duplicated tweets, tweets that contain mixed fake and genuine news, and tweets where the fake news was meant as sarcasm. Statistical information about the manually annotated corpus is shown in the following table:</br>

| | Fake Tweets | Not Fake Tweets |
| --- | --- | --- |
| Total Tweets| 835 | 702 |
| Total Words | 20,395 | 19,852 |
| Unique Words| 6,246 | 7,115 |
| Total Characters| 117,630 | 113,121 |

2.	Automatic Annotated Corpus: </br>
We trained different machine learning classifiers on the manually annotated corpus and used the best performing classifier to automatically predict the fake news classes of remaining unlabeled tweets. The outcome of the prediction process is 34,529 tweets (19,582 fake and 19,582 genuine) as shown in the following table. </br>

| | Fake Tweets | Not Fake Tweets |
| --- | --- | --- |
| Total Tweets| 19,582 | 14,947 |
| Total Words | 479,349 | 463,768 |
| Unique Words| 79,383 | 88,037 |
| Total Characters| 2,855,454 | 2,680,067 |


<b> Machine Learning Classifiers:</b></br>
Six machine learning classifiers were used to perform fake news classification for both datasets: Naïve Bayes [19], Logistic Regression (LR), Support Vector Machine (SVM), Multilayer Perceptron (MLP), Random Forest Bagging Model (RF), and eXtreme Gradient Boosting Model (XGB). The following are the hyper-parameters used with each classifier:</br>
•	NB: alpha=0.5</br>
•	LR: with default values</br>
•	SVM: c=1.0, kernel=linear, gamma=3</br>
•	MLP: activation function=ReLU, maximum iterations=30, learning rate=0.1</br>
•	RF: with default values</br>
•	XGB: with default values</br>

<b> Citations:</b></br>
@article{mahlous2021fake,</br>
  title={Fake news detection in Arabic tweets during the COVID-19 pandemic},</br>
  author={Mahlous, Ahmed Redha and Al-Laith, Ali},</br>
  journal={Int J Adv Comput Sci Appl},</br>
  year={2021}</br>
}

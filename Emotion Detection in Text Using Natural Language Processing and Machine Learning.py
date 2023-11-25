#!/usr/bin/env python
# coding: utf-8

# In[101]:


"""
Title: "Emotion Detection in Text Using Natural Language Processing and Machine Learning"

>Text classification
>Sentiment Analysis


Aim:
The goal of the Emotion Analysis from Text project is to teach the computer to understand and recognize emotions in written text. We want the computer to read sentences and figure out if the writer is happy, sad, angry, or feeling other emotions. This can be useful for things like understanding customer reviews or analyzing social media posts.

Author: Dharmagna Vyas
"""


# In[ ]:


### Step 1: Import necessary libraries


# In[14]:


import pandas as pd
import numpy as np


# In[15]:


#Load Data Viz Pkgs
import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


get_ipython().system('pip install neattext')


# In[17]:


### Step 2: Load Text Cleaning Library
import neattext.functions as nfx


# In[18]:


# Step 3: Load the dataset
df = pd.read_csv("C:\\Users\\Dharmagna Vyas\\Downloads\\emotion_dataset.csv\\emotion_dataset.csv")


# In[19]:


# Step 4: Display the first few rows of the dataset
df.head()


# In[20]:


# Step 5: Display dataset shape and data types
df.shape


# In[21]:


df.dtypes


# In[22]:


# Step 6: Check for missing values
df.isnull().sum()


# In[23]:


#Value Counts of Emotions
df['Emotion'].value_counts()  


# In[24]:


df['Emotion'].value_counts().plot(kind='bar') 


# In[25]:


# Step 7: Visualize the distribution of emotions
plt.figure(figsize = (20,10))
sns.countplot(x='Emotion',data=df)
plt.show()


# In[26]:


# Step 8: Sentiment Analysis using TextBlob


# In[27]:


get_ipython().system('pip install TextBlob')


# In[28]:


#Sentiment Analysis
from textblob import TextBlob


# In[29]:


def get_sentiment(text):
    blob= TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        result = "Positive"
    elif sentiment < 0:
        result = "Negative"
    else:
        result="Neutral"
    return result


# In[30]:


#TestFxn
get_sentiment("I love coding")


# In[31]:


# Step 9: Apply sentiment analysis to the 'Text' column
df['Sentiment'] = df['Text'].apply(get_sentiment)


# In[32]:


df.head()


# In[33]:


# First method:using matplotlib
# Compare emotion vs sentiment
df.groupby(['Emotion','Sentiment']).size().plot(kind = 'bar')


# In[34]:


# Step 10: Visualize the relationship between Emotion and Sentiment
#for better view
import seaborn as sns
import matplotlib.pyplot as plt
sns.catplot(x="Emotion", hue='Sentiment', kind='count', data=df, aspect = 1.5)
plt.show()


# In[35]:


### Text Cleaning
#Remove noise
#Stop words
#Special characters
#punctuations
#emojis


# In[36]:


dir(nfx)


# In[37]:


# Step 11: Text Cleaning using NeatText


# In[38]:


df['Clean_Text']=df['Text'].apply(nfx.remove_stopwords)


# In[39]:


df['Clean_Text']=df['Text'].apply(nfx.remove_userhandles)


# In[40]:


df['Clean_Text']=df['Text'].apply(nfx.remove_punctuations)


# In[41]:


df[['Text','Clean_Text']]


# In[42]:


#Keyword Extraction
#extract most common words per class of emotion


# In[43]:


from collections import Counter


# In[44]:


# Step 12: Explore Keywords for each emotion
emotion_list =df['Emotion'].unique().tolist()


# In[45]:


emotion_list 


# In[46]:


# Step 13: Function to extract keywords
def extract_keywords(text,num=50):
    tokens=[tok for tok in text.split()]
    most_common_tokens = Counter(tokens).most_common(num)
    return dict(most_common_tokens)


# In[47]:


joy_list = df[df['Emotion'] == 'joy']['Clean_Text'].tolist()


# In[48]:


#joy document
joy_docx = ' '.join(joy_list)


# In[49]:


joy_docx


# In[50]:


#extract keywords
keyword_joy = extract_keywords(joy_docx)


# In[51]:


keyword_joy


# In[52]:


#plot
def plot_most_common_words(mydict,emotion_name):
    df_01 = pd.DataFrame(mydict.items(),columns=['token','count'])
    plt.figure(figsize=(20,10))
    plt.title("Plot of {} Most common keywords".format(emotion_name))
    sns.barplot(x='token',y='count',data=df_01)
    plt.xticks(rotation=45)
    plt.show()


# In[53]:


plot_most_common_words(keyword_joy,"Joy")


# In[54]:


surprise_list = df[df['Emotion'] == 'surprise']['Clean_Text'].tolist()
#joy document
surprise_docx = ' '.join(surprise_list)
#Extract Keywords
#extract keywords
keyword_surprise = extract_keywords(surprise_docx)


# In[55]:


plot_most_common_words(keyword_surprise,"Surprise")


# In[56]:


emotion_list 


# In[57]:


neutral_list = df[df['Emotion'] == 'neutral']['Clean_Text'].tolist()
#joy document
neutral_docx = ' '.join(neutral_list)
#Extract Keywords
#extract keywords
keyword_neutral = extract_keywords(neutral_docx)


# In[58]:


plot_most_common_words(keyword_neutral,"Neutral")


# In[59]:


# Step 14: Loop through emotions and create keyword plots


# In[60]:


sadness_list = df[df['Emotion'] == 'sadness']['Clean_Text'].tolist()
#joy document
sadness_docx = ' '.join(sadness_list)
#Extract Keywords
keyword_sadness = extract_keywords(sadness_docx)
plot_most_common_words(keyword_sadness,"sadness")

fear_list = df[df['Emotion'] == 'fear']['Clean_Text'].tolist()
#joy document
fear_docx = ' '.join(fear_list)
#Extract Keywords
keyword_fear = extract_keywords(fear_docx)
plot_most_common_words(keyword_fear,"fear")

anger_list = df[df['Emotion'] == 'anger']['Clean_Text'].tolist()
#joy document
anger_docx = ' '.join(anger_list)
#Extract Keywords
keyword_anger = extract_keywords(anger_docx)
plot_most_common_words(keyword_anger,"anger")

shame_list = df[df['Emotion'] == 'shame']['Clean_Text'].tolist()
#joy document
shame_docx = ' '.join(shame_list)
#Extract Keywords
keyword_shame = extract_keywords(shame_docx)
plot_most_common_words(keyword_shame,"shame")

disgust_list = df[df['Emotion'] == 'disgust']['Clean_Text'].tolist()
#joy document
disgust_docx = ' '.join(disgust_list)
#Extract Keywords
keyword_disgust= extract_keywords(disgust_docx)
plot_most_common_words(keyword_disgust,"disgust")


# In[61]:


get_ipython().system('pip install wordcloud')


# In[62]:


# Step 15: Generate Word Clouds
from wordcloud import WordCloud


# In[63]:


def plot_wordcloud(docx):
    mywordcloud = WordCloud().generate(docx)
    plt.figure(figsize=(20,10))
    plt.imshow(mywordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.show()


# In[64]:


plot_wordcloud(joy_docx)


# In[65]:


plot_wordcloud(surprise_docx)


# In[66]:


# Step 16: Machine Learning - Naive Bayes Classifier


# In[67]:


pip install --upgrade scikit-learn


# In[68]:


get_ipython().system('pip install --upgrade scikit-learn')


# In[69]:


import sklearn
print(sklearn.__version__)


# In[70]:


pip install --upgrade scikit-learn


# In[71]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[72]:


#Split Dataset
from sklearn.model_selection import train_test_split


# In[73]:


#Build Features from our text
Xfeatures =df['Clean_Text']
ylabels = df['Emotion']


# In[74]:


Xfeatures


# In[75]:


# Step 17: Vectorize the text data
cv = CountVectorizer()
cv.fit_transform(Xfeatures)
X = cv.fit_transform(Xfeatures)


# In[76]:


list(cv.get_feature_names_out())


# In[77]:


# Step 18: Split the dataset
X_train, X_test, y_train,y_test = train_test_split(X,ylabels,test_size=0.3,random_state = 42)


# In[78]:


# Step 19: Build Naive Bayes Model
nv_model = MultinomialNB()
nv_model.fit(X_train,y_train)


# In[79]:


# Step 20: Evaluate the Naive Bayes model
#Accuracy
nv_model.score(X_test, y_test)


# In[80]:


#Predictions
y_pred_for_nv = nv_model.predict(X_test)


# In[81]:


y_pred_for_nv


# In[82]:


#Make a single prediction
#Vectorized out text
#applied our model


# In[83]:


sample_text = ['I love this so much']


# In[84]:


vect = cv.transform(sample_text).toarray()


# In[85]:


#Make prediction
nv_model.predict(vect)


# In[86]:


# Check for the prediction probablity(Percentage)/cofidence score
nv_model.predict_proba(vect)


# In[87]:


#Get all class for our model
nv_model.classes_


# In[88]:


np.max(nv_model.predict_proba(vect))


# In[89]:


def predict_emotion(sample_text,model):
    myvect =  cv.transform(sample_text).toarray()
    prediction = model.predict(myvect)
    pred_proba = model.predict_proba(myvect)
    pred_percentage_for_all = dict(zip(model.classes_,pred_proba[0]))
    print("Prediction:{},Prediction Score:{}".format(prediction[0],np.max(pred_proba)))
    return pred_percentage_for_all   


# In[90]:


predict_emotion(sample_text,nv_model)


# In[91]:


predict_emotion(["He hates running"], nv_model)


# In[92]:


#model evaluation
#classification
print(classification_report(y_test,y_pred_for_nv))


# In[93]:


#confusioin matrix
confusion_matrix(y_test, y_pred_for_nv)


# In[94]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# model evaluation
print(classification_report(y_test, y_pred_for_nv))

# confusion matrix
cm = confusion_matrix(y_test, y_pred_for_nv)

# custom confusion matrix plot
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, linewidths=.5, square=True,
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# get unique class labels
classes = df['Emotion'].unique()

# plot confusion matrix
plot_confusion_matrix(cm, classes, title='Confusion Matrix')


# In[95]:


# Step 21: Save Naive Bayes model
import joblib


# In[96]:


model_file = open("emotion_classifier_nv_model_25.pkl","wb")
joblib.dump(nv_model , model_file)
model_file.close()


# In[97]:


# Step 22: Machine Learning - Logistic Regression


# In[98]:


#Logistic Regrssion model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)


# In[99]:


# Step 23: Evaluate Logistic Regression model
#Accuracy
lr_model.score(X_test,y_test)


# In[100]:


# Step 24: Save Logistic Regression model
model_file_lr = open("emotion_classifier_lr_model.pkl", "wb")
joblib.dump(lr_model, model_file_lr)
model_file_lr.close()


# In[105]:


#Simple prediction
predict_emotion(sample_text,lr_model)


# In[123]:


# Step 25: Cross-validated Accuracy
from sklearn.model_selection import cross_val_score
# Assuming your model is named 'lr_model' and X, y are your feature matrix and target variable
scores = cross_val_score(lr_model, X_test, y_test, cv=5)  # 5-fold cross-validation
print("Cross-validated Accuracy:", scores.mean())


# In[124]:


# Step 26: Model Evaluation Metrics

from sklearn.metrics import confusion_matrix, classification_report
# Assuming 'lr_model' is your Logistic Regression model
y_pred = lr_model.predict(X_test)

# Step 27: Confusion Matrix for Logistic Regression

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report)


# In[126]:


# Step 28: Additional Model Evaluation Metrics for Logistic Regression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming 'lr_model' is your Logistic Regression model
y_pred = lr_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)


# In[102]:


"""
What I Learned:
Through this project, I learned how to make computers understand emotions in text. I learned to clean up messy text data, analyze sentiments, and use machine learning to predict emotions. I also learned how to evaluate how well the computer is doing and save my work for future use. It's like teaching a computer to read and understand feelings in written words!
"""


# In[ ]:





# In[ ]:





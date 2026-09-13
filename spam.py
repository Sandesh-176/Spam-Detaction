import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

data = pd.read_csv(r"C:\Users\Sandesh\OneDrive\Desktop\Spam Detaction new\spam.csv", encoding='latin-1')

data=data[['v1','v2']]
data.columns=['Category','Message']

data.drop_duplicates(inplace=True)
data['Category']=data['Category'].replace(['ham','spam'],['Not Spam','Spam'])

#Input and Output
mess = data['Message']
cat = data['Category']

(mess_train, mess_test, cat_train, cat_test) = train_test_split(mess, cat, test_size=0.2 ) #split data into training and testing data

cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

# creating Model
model = MultinomialNB()
model.fit(features, cat_train) #Train the model with traning (input) data

#Test our Model
features_test = cv.transform(mess_test)
# print(model.score(features_test, cat_test)) #Test the model with testing data and print the accuracy score

#Predict the Data
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result

#Web Application
st.header("Spam Detection")

input_mess = st.text_input("Enter Message Here")

if st.button('Validate'):
    output = predict(input_mess)
    st.markdown(output)



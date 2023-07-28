import streamlit as st
import pickle
from DecisionTree import vectorize, TreeClassifier, Node
from difflib import get_close_matches
import streamlit_tags as st_tags
import pickle

# Loading set of possible symptoms that our model uses
with open("columns.txt", "rb") as file:
    columns = pickle.load(file)

# Loading Decision tree classifier model
with open("DecisionTreeModel.txt", "rb") as file:
    model = pickle.load(file)

# Loading all possible symptoms
with open("available_symptoms.txt", "rb") as file:
    available_symptoms = pickle.load(file)

available_symptoms_keys = [*available_symptoms.keys()]
available_symptoms_values = [*available_symptoms.values()]

# Creating input columns implemented with tag elements
temp_symptoms = st_tags.st_tags(
    label="# Enter your symptoms:",
    text="Press enter to add more",
    suggestions=available_symptoms_keys,
    maxtags=-1,
    key=None,
)

# In case where wrong spelling for the symptoms were given, following for loop will replace wrongly spelled symptom with the actually existing symptom that is spelled almost the same way
symptoms = []
for x in temp_symptoms:
    matches = get_close_matches(x, available_symptoms_keys, cutoff=0.3) # finding all matches for the wrongly spelled symptom that is atleast 30% similar to it
    if len(matches):
        temp = matches[0]
    else:
        temp = x

    symptoms.append(available_symptoms[temp])

vector = vectorize(symptoms, columns) # Vectorizing symptoms so that it could be fed to the model

if st.button("Diagnose"): # Model will give its prediction after the click of this word
    predicted = model.predict([vector])
    st.text(predicted[0])
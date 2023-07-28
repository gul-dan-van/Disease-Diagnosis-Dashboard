DS 250
Assignment 1c

Name: Gauranshu
ID: 12040560

DecisionTree.py:
    This script used to form the base of our Decision Tree model. It also loads the data in a more useful format, vectorizes the symptoms so that they could be used in a more mathematical way.
    Accuracy of the Tree formed with depth 10 and min samples per leaf 5 was observed as 94%
    This script also writes two pickle files, 'DecisionTreeModel.txt' and 'columns.txt'

scrapper.py:
    This script scraps thesaurus website for the synonyms of the symptoms that our Tree model considers. It will form a dictionary and all synonyms of a particular symptoms will be a key of this dictionary, all holding actual symptoms used in the model as their value.
    This script will dump this dictionary into a pickle file.

App.py:
    This script allows the user to give their symptoms as input, and on a press of button, it will return you the most probable disease you might have, all on a GUI formed with the help "streamlit" library.
    When the users will be typing symptoms, script will suggest the word users might be typing, if it guesses correctly, they can press tab for autocompletion of the symptom and press enter to form a tag containing the symptom. The tag can be removed with a click on a cross on the right of the tag. If the users pressed enter and formed a tag of that symptom even though the script wasn't able to guess what they were typing, or it might be a case of wrong spelling, script will match that unknown word with some other existing symptom that it resembles the most with.
    On the click of "diagnose" button, script will format and send the symptoms to the tree model and show users the diagnosed disease.

available_symptoms.txt:
    It holds a dictionary that will match symptoms to a synonym of the same that exists in the Tree model formed

columns.txt:
    It holds the list of symptoms that are used in the Tree model

DecisionTreeModel.txt:
    It hold the final trained Tree model.

Disease.pdf:
    It is a visualisation of model formed with the help of sklearn library.
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import re
import time
import os
import pickle

# Loading list of symptoms
with open("columns.txt", "rb") as file:
    columns = pickle.load(file)

CWD = os.getcwd()
DRIVER_PATH = "C:\\Program Files (x86)\\chromedriver.exe"
DUMMY_URL = r"https://www.thesaurus.com/browse/REGEX" # We will replace REGEX word with the word that we would want to search
url_edit_re = re.compile("REGEX")

# Creating an instance of chromedriver
options = Options()
options.add_argument("--diable-extensions")
options.add_argument("--headless")
driver = webdriver.Chrome(DRIVER_PATH, options=options)

# Formatting words used for symptoms
available_symptoms = {}
for symptom in columns:
    temp = " ".join([x[0].upper() + x[1:] for x in symptom.split("_")])
    available_symptoms[temp] = symptom

SYMPTOMS = [*available_symptoms.keys()]
for sym in SYMPTOMS: # Iterating over all symptoms to find there synonyms from web
    sym = "%20".join(sym.split(" ")) # Formating word used for symptom to fit in the way that thesaurus website uses in its URLs
    URL = url_edit_re.sub(sym, DUMMY_URL) # replacing REGEX in dummy url with symptom name

    driver.get(URL) # Getting the website at the formated url
    time.sleep(5)

    try:
        Xpath = "/html/body/div[1]/div[2]/div/div/div[2]/main/section/section/div[2]/div[2]/ul" # XPATH to the synonym table element
        synonyms = driver.find_element_by_xpath(Xpath) # Getting the synonym table
        for word in synonyms.text.split("\n"):
            available_symptoms[word] = available_symptoms[sym]

    except:
        continue

driver.quit()

# Loading possible symptoms in a pickle file
with open("available_symptoms.txt", "wb") as file:
    pickle.dump(available_symptoms, file)

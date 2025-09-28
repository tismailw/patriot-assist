"""
Resources: 
https://www.youtube.com/watch?v=X2vAabgKiuM&t=521s    
https://www.nltk.org/
"""

import nltk
from nltk.corpus import brown
from nltk.stem import WordNetLemmatizer
import json

JSONL_PATH = "gmu_cs_courses_with_prereqs.jsonl"

#does not work bc  we're trying to tokenize a object
#print(word_tokenize(JSONL_PATH)) 

with open(JSONL_PATH, 'r', encoding='utf-8') as file:
    first_line = file.readline()
    data = json.loads(first_line)
#print(data)

#tokenize each word in course_name and course_summary
from nltk.tokenize import word_tokenize
text = data.get('course_name', "") + " " + data.get("course_summary", "")
tokenized = word_tokenize(text)
print(tokenized[:50])

#find the frequency of each word
from nltk.probability import FreqDist
fdist = FreqDist()

for word in tokenized:
    fdist[word.lower()] +=1

print(dict(fdist))
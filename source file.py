# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 21:31:34 2025

@author: AjayBarath
"""

import pandas as pd
import re

# Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    return text

data['cleaned'] = data['message'].apply(clean_text)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['cleaned'])
y = data['label']

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))

def classify_email(text):
    text = clean_text(text)
    vect = vectorizer.transform([text])
    prediction = model.predict(vect)[0]
    return "Spam Folder" if prediction == "spam" else "Inbox"

# Example
email = "Congratulations! You've won a free ticket. Click here!"
print("Classification Result:", classify_email(email))


# Simulated feedback
new_text = "Please review the attached invoice."
correct_label = "ham"

# Append to dataset
new_row = pd.DataFrame([[correct_label, new_text, clean_text(new_text)]], columns=data.columns)
data = pd.concat([data, new_row], ignore_index=True)

# Re-vectorize and retrain
X = vectorizer.fit_transform(data['cleaned'])
y = data['label']
model.fit(X, y)


import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])

true_ham, false_spam = cm[0]
false_ham, true_spam = cm[1]

labels = ['Ham', 'Spam']
correct = [true_ham, true_spam]
incorrect = [false_ham, false_spam]

x = np.arange(len(labels))
width = 0.5

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(x, correct, width, label='Correct', color='mediumseagreen')
ax.bar(x, incorrect, width, bottom=correct, label='Incorrect', color='tomato')

ax.set_ylabel('Email Count')
ax.set_title('Correct vs Incorrect Classifications')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.savefig("stacked_accuracy_chart.png")
plt.show()



import matplotlib.pyplot as plt

# Count predictions
spam_count = sum(y_pred == 'spam')
ham_count = sum(y_pred == 'ham')

labels = ['Ham (Inbox)', 'Spam (Spam Folder)']
sizes = [ham_count, spam_count]
colors = ['skyblue', 'salmon']
explode = (0.05, 0.1)

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140, explode=explode)
plt.title('Spam vs Ham Email Classification')
plt.axis('equal')
plt.savefig("spam_ham_piechart.png")
plt.show()



from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define the class labels in correct order
labels = ['ham', 'spam']

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=labels)

# Plot heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)

plt.title("Confusion Matrix - Spam Detection")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")  # Save as image
plt.show()
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer

# preprocessing
# It is the one preprocessed and stored in local
fake = pd.read_csv('dataset/Fake_.csv')
true = pd.read_csv('dataset/True_.csv')
concat2 = pd.concat([fake, true])

# find the top words
vect = TfidfVectorizer()
X = vect.fit_transform(concat2['text'])
y = concat2['is_fake']
topWords = []
s = SelectKBest(chi2, k=3)
X_new = s.fit_transform(X, y)
mask = s.get_support()
for bool, feature in zip(mask, vect.get_feature_names()):
    if bool:
        topWords.append(feature)
print(topWords)

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

print("TF: ")
vectorizer = HashingVectorizer()
train_data = vectorizer.fit_transform(train['data'])
test_data = vectorizer.fit_transform(test['data'])

metrics = ['euclidean', 'cosine']

for metric in metrics:
    for k in [1, 5, 10]:
        classifier = KNeighborsClassifier(n_neighbors=k, metric=metric)
        classifier.fit(train_data, train['target'])
        pred = classifier.predict(test_data)

        acc = accuracy_score(pred, test['target'])

        print(metric)
        print(' k = {} | '.format(k) + 'Accuracy: {}'.format(acc))

print("TFIDF")
vectorizer = TfidfVectorizer()
train_data = vectorizer.fit_transform(train['data'])
test_data = vectorizer.fit_transform(test['data'])

metrics = ['euclidean', 'cosine']

for metric in metrics:
    for k in [1, 5, 10]:
        classifier = KNeighborsClassifier(n_neighbors=k, metric=metric)
        classifier.fit(train_data, train['target'])
        pred = classifier.predict(test_data)

        acc = accuracy_score(pred, test['target'])

        print(metric)
        print(' k = {} | '.format(k) + 'Accuracy: {}'.format(acc))
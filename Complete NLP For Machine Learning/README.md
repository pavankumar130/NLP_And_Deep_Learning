# Natural Language Processing (NLP) Concepts and Code 🧠

This repository is a practical guide to fundamental concepts in Natural Language Processing (NLP). It provides clear explanations and runnable Python code examples for each topic, helping you understand how these techniques are applied in practice.

-----

## 📑 Index

1. [Introduction to NLP](#-natural-language-processing-nlp)
2. [Tokenization](#1-tokenization)
3. [Stemming](#2-stemming)
4. [Lemmatization](#3-lemmatization)
5. [Stopwords](#4-stopwords)
6. [Bag of Words (BoW)](#5-bag-of-words-bow)
7. [TF-IDF](#6-tf-idf)
8. [Word2Vec](#7-word2vec)
9. [AvgWord2Vec](#8-avgword2vec)

-----

## 📚 Natural Language Processing (NLP)

Natural Language Processing is a subfield of artificial intelligence that focuses on the interaction between computers and human language. The goal is for computers to process, understand, and generate human language in a valuable way.

### Topics

### 1\. Tokenization

**Tokenization** is the process of breaking down a text into smaller units called **tokens**. These tokens can be words, phrases, or even individual characters. It is typically the first step in an NLP pipeline, as it structures the raw text for further analysis.

#### Sample Python Code

We can use the `nltk` library for tokenization. You might need to install it first (`pip install nltk`) and download the necessary data (`nltk.download('punkt')`).

```python
import nltk
from nltk.tokenize import word_tokenize

text = "Natural language processing is a fascinating field."
tokens = word_tokenize(text)

print(tokens)
```

**Explanation:**
The `word_tokenize` function splits the input string into a list of words and punctuation marks, treating each as a separate token. The output will be `['Natural', 'language', 'processing', 'is', 'a', 'fascinating', 'field', '.']`.

### 2\. Stemming

**Stemming** is a technique that reduces words to their **root or base form** by removing suffixes. The goal is to group together words with similar meanings. Stemming is a heuristic process and may not always produce a valid dictionary word. For example, "running" and "ran" might both be stemmed to "runn".

#### Sample Python Code

We use the Porter Stemmer from the `nltk` library.

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ["running", "runner", "runs", "ran"]

stemmed_words = [stemmer.stem(word) for word in words]

print(stemmed_words)
```

**Explanation:**
The code initializes a `PorterStemmer` object and then applies the `stem()` method to each word in the list. The output will be `['run', 'runner', 'run', 'ran']`. Notice how "running" and "runs" are reduced to 'run', while 'runner' and 'ran' are not.

### 3\. Lemmatization

**Lemmatization** is a more sophisticated process than stemming. It reduces words to their **lemma**, which is the canonical or dictionary form of a word. Unlike stemming, it uses vocabulary and morphological analysis, ensuring the resulting root word is a valid word.

#### Sample Python Code

We use the WordNet Lemmatizer from `nltk`. You might need to download the 'wordnet' data (`nltk.download('wordnet')`).

```python
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()
words = ["running", "runner", "runs", "ran", "better"]

lemmatized_words = [lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in words]

print(lemmatized_words)
```

**Explanation:**
The code initializes a `WordNetLemmatizer`. We specify `pos=wordnet.VERB` to treat the words as verbs, which is crucial for correct lemmatization (e.g., "running" becomes "run"). Note how "better" would be lemmatized to "good" if treated as an adjective.

### 4\. Stopwords

**Stopwords** are common words that often carry little significant meaning and can be filtered out to improve the performance of NLP models. Examples include "the," "a," "is," and "in." Removing them helps reduce noise and focus on more important terms.

#### Sample Python Code

We use the stopwords corpus from `nltk`. You might need to download it first (`nltk.download('stopwords')`).

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

text = "This is a sample sentence to demonstrate the removal of stopwords."
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(text)

filtered_sentence = [w for w in word_tokens if w.lower() not in stop_words]

print("Original Sentence:", text)
print("Filtered Sentence:", ' '.join(filtered_sentence))
```

**Explanation:**
The code first gets the set of English stopwords. It then tokenizes the input text and iterates through the tokens, keeping only those that are not in the stopwords set. The output removes words like "is", "a", "to", "the", and "of".

### 5\. Bag of Words (BoW)

The **Bag of Words (BoW)** model is a simple way of representing text data for use in machine learning. It describes the occurrence of words within a document, ignoring grammar and word order. It involves creating a vocabulary of all unique words and then representing each document as a vector showing the count of each word.

#### Sample Python Code

We use `CountVectorizer` from `scikit-learn` to create a BoW model.

```python
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "I love deep learning.",
    "Deep learning is a subset of machine learning."
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("BoW Matrix:\n", X.toarray())
```

**Explanation:**
`CountVectorizer` automatically tokenizes the documents and builds a vocabulary. It then creates a matrix where each row represents a document, and each column represents a word from the vocabulary. The cell value is the word's frequency in that document.

-----

### 6\. TF-IDF

**TF-IDF** (**Term Frequency-Inverse Document Frequency**) is a statistical measure used to evaluate how important a word is to a document in a collection. It assigns a higher value to words that are frequent in a specific document but rare in the overall corpus, thus highlighting words that are more unique and informative.

#### Sample Python Code

We use `TfidfVectorizer` from `scikit-learn` for this task.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "The cat sat on the mat.",
    "The dog sat on the log.",
    "Cats and dogs are pets."
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("TF-IDF Matrix:\n", X.toarray())
```

**Explanation:**
`TfidfVectorizer` computes the TF-IDF score for each word in each document. The resulting matrix shows the importance of each word. Notice that common words like "the" will have a low TF-IDF score because they appear in multiple documents.

### 7\. Word2Vec

**Word2Vec** is a popular technique for learning **word embeddings**, which represent words as dense, real-valued vectors. Words with similar meanings are located closer to each other in this vector space. It is a powerful method for capturing semantic relationships between words.

#### Sample Python Code

We use the `Gensim` library. You might need to install it first (`pip install gensim`).

```python
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize

text = "The quick brown fox jumps over the lazy dog. The dog is a pet."
sentences = [word_tokenize(sentence.lower()) for sentence in nltk.sent_tokenize(text)]

model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, workers=4)

# Get the vector for the word 'dog'
dog_vector = model.wv['dog']
print("Vector for 'dog':\n", dog_vector)

# Find words similar to 'dog'
similar_words = model.wv.most_similar('dog', topn=2)
print("\nWords similar to 'dog':", similar_words)
```

**Explanation:**
The code first tokenizes the text into a list of sentences, where each sentence is a list of words. The `Word2Vec` model is then trained on these sentences. `vector_size` determines the dimension of the word vectors, and `window` is the maximum distance between the current and predicted word. The output shows the vector for "dog" and other words with similar vector representations, like "pet" or "lazy".

### 8\. AvgWord2Vec

**AvgWord2Vec** is a simple yet effective method for creating document-level representations. It involves taking the average of the Word2Vec vectors for all the words in a document. This results in a single vector that represents the entire document, which can then be used for tasks like document classification.

#### Sample Python Code

This is a manual implementation using a pre-trained Word2Vec model.

```python
import numpy as np

# Let's use the model from the previous example
# Assume 'model' is the trained Word2Vec model from above
# and 'sentences' is the list of tokenized sentences

def document_vector(doc_tokens, model):
    """Calculates the average vector for a document."""
    # Filter out words not in the model's vocabulary
    vectors = [model.wv[word] for word in doc_tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

document1_vec = document_vector(sentences[0], model)
document2_vec = document_vector(sentences[1], model)

print("Vector for first document:\n", document1_vec)
print("\nVector for second document:\n", document2_vec)
```

**Explanation:**
The `document_vector` function takes a list of tokens (a document) and the Word2Vec model. It retrieves the vector for each word in the document and then calculates the average of these vectors. This creates a fixed-size vector representation for the entire document, regardless of its length.
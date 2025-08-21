import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,  confusion_matrix
from sklearn.utils import resample
import spacy
from sklearn.pipeline import FeatureUnion
import seaborn as sns
import matplotlib.pyplot as plt



# Get trained pipelines for German -> for PoS tagging later
nlp = spacy.load('de_core_news_sm')


# Function to extract POS bigrams from text
def extract_pos_bigrams(text):
    doc = nlp(text)
    pos_tags = [token.pos_ for token in doc]
    pos_bigrams = [" ".join(pos_tags[i:i+2]) for i in range(len(pos_tags)-1)]

    return " ".join(pos_bigrams)


# Main path data
data = r"C:/Users/whatt/iCloudDrive/UZH/Master/HS23/CL/Computational Forensic Linguistics/Term Paper/Twitter Data/"

# Load CSV files for each politician on Twitter
df_author1 = pd.read_csv(f'{data}Beatrix_vStorch_0104_0110.csv', encoding='utf-8')
df_author2 = pd.read_csv(f'{data}FraukePetry_0104_0110.csv', encoding='utf-8')
df_author3 = pd.read_csv(f'{data}SWagenknecht_0104_0110.csv', encoding='utf-8')



# Splitting data in 80/20 training/test data for each author
train_author1, test_author1 = train_test_split(df_author1, test_size=0.2, random_state=42)
train_author2, test_author2 = train_test_split(df_author2, test_size=0.2, random_state=42)
train_author3, test_author3 = train_test_split(df_author3, test_size=0.2, random_state=42)

# Address class imbalance to the size of the bigger class
df_oversampled_2 = resample(train_author2, replace=True, n_samples=len(train_author1), random_state=42)
df_oversampled_3 = resample(train_author3, replace=True, n_samples=len(train_author1), random_state=42)

# Concatenate the training and testing sets for each author
train_data = pd.concat([train_author1, df_oversampled_2,  df_oversampled_3], ignore_index=True)
test_data = pd.concat([test_author1, test_author2, test_author3], ignore_index=True)

# Shuffle the concatenated dataframes
train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)


# Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define classifiers
classifiers = {
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(kernel='linear'),
    'Logistic Regression': LogisticRegression()
}



# Feature extraction for unigrams and TF-IDF
feature_sets = {
    #  fit_transform udn transform weg
    "Word 4-grams": CountVectorizer(analyzer='word', ngram_range=(4, 4)),
    "POS Bigrams": lambda l: l.apply(extract_pos_bigrams),
    "Character 4-grams": CountVectorizer(analyzer='char', ngram_range=(4, 4)),
    "Tf-Idf": TfidfVectorizer(analyzer='word', ngram_range=(1, 1)),
    "All features": FeatureUnion([
        # add functions
        ("POS Bigrams", extract_pos_bigrams),
        ("Word 4-grams", CountVectorizer(analyzer='word', ngram_range=(4, 4))),
        ("Character 4-grams", CountVectorizer(analyzer='char', ngram_range=(4, 4))),
        ("Tf-Idf", TfidfVectorizer(analyzer='word', ngram_range=(1, 1)))
    ])
}

# word 4 grams
word4grams = CountVectorizer(analyzer='word', ngram_range=(4, 4))

# pos bigrams 
pos_bigrams_vectoriser = CountVectorizer(analyzer='word', ngram_range=(2, 2))

# Character 4-grams
char4grams = CountVectorizer(analyzer='char', ngram_range=(4, 4))

# Tf-Idf
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1))

# All features
all_features = FeatureUnion([
    ("Word 4-grams", word4grams),
    ("POS Bigrams", pos_bigrams_vectoriser),
    ("Character 4-grams", char4grams),
    ("Tf-Idf", tfidf)
])

results = {}

print(train_data['Content'])

for classifier_name, classifier in classifiers.items():
    results[classifier_name] = {}
    for feature_set_name, feature_set in feature_sets.items():
        # Initialise confusion matrix
        trial_confusion_matrix = np.zeros((3, 3), dtype=int)

        results[classifier_name][feature_set_name] = {'raw': [], 'cf_raw': [], 'mean': 0.0, 'std': 0.0}

        fold = 1
        for train_index, val_index in kf.split(train_data['Content'], train_data['Author']):

            # Split the training data into training and validation sets
            X_train, X_val = train_data['Content'].iloc[train_index], train_data['Content'].iloc[val_index]
            y_train, y_val = train_data['Author'].iloc[train_index], train_data['Author'].iloc[val_index]
            

            # Extract features from the training data

            # Feature extraction for Word 4-grams
            if feature_set_name == "Word 4-grams":
                X_train_transformed = word4grams.fit_transform(X_train)
                X_val_transformed = word4grams.transform(X_val)
            
            # Feature extraction for POS Bigrams
            elif feature_set_name == "POS Bigrams":
                X_train_transformed = pos_bigrams_vectoriser.fit_transform(
                    extract_pos_bigrams(text) for text in X_train
                )
                X_val_transformed = pos_bigrams_vectoriser.transform(
                    extract_pos_bigrams(text) for text in X_val
                )
               
            # Feature extraction for Character 4-grams
            elif feature_set_name == "Character 4-grams":
                X_train_transformed = char4grams.fit_transform(X_train)
                X_val_transformed = char4grams.transform(X_val)
               
            # Feature extraction for Tf-Idf
            elif feature_set_name == "Tf-Idf":
                X_train_transformed = tfidf.fit_transform(X_train)
                X_val_transformed = tfidf.transform(X_val)

            # Feature extraction for All features
            elif feature_set_name == "All features":
                X_train_transformed = all_features.fit_transform(X_train)
                X_val_transformed = all_features.transform(X_val)
           

            # Train the classifier on the training data
            classifier.fit(X_train_transformed, y_train)

            # Use the trained classifier to make predictions on the validation data
            y_val_pred = classifier.predict(X_val_transformed)

            # Evaluate the classifier's performance
            accuracy = accuracy_score(y_val, y_val_pred)

            # Confusion Matrix for each fold
            confusion_matrix_val = confusion_matrix(y_val, y_val_pred)
            trial_confusion_matrix += confusion_matrix_val
            

            # Store the results
            results[classifier_name][feature_set_name]['raw'].append(accuracy)

            print(f'Classifier: {classifier_name}, Feature Set: {feature_set_name}, Accuracy: {accuracy}')
            
            fold += 1
        
        # Calculate mean and standard deviation of the accuracy for each classifier and feature set
        results[classifier_name][feature_set_name]['mean'] = np.mean(results[classifier_name][feature_set_name]['raw'])
        results[classifier_name][feature_set_name]['std'] = np.std(results[classifier_name][feature_set_name]['raw'])
        
        # Confusion Matrix for each classifier and feature set (summation of all folds)
        cm_df = pd.DataFrame(trial_confusion_matrix, index=['Beatrix_vStorch', 'FraukePetry', 'SWagenknecht'], columns=['Beatrix_vStorch', 'FraukePetry', 'SWagenknecht'])

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_df, annot=True)
        plt.title(f'Confusion Matrix for {classifier_name} with {feature_set_name}')
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        plt.show()

# End results
print(results)






 
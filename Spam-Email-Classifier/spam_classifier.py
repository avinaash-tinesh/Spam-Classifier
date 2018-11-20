import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter
import sklearn
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import string


def load_lingspam_data(init_folder):
	folders = [folder.path for folder in os.scandir(init_folder) if os.path.isdir(folder)]

	test_set, test_set_classification = lingspam_scraper(folders[0])
	training_set, training_set_classification = lingspam_scraper(folders[1])

	return test_set, training_set, test_set_classification, training_set_classification

	

def lingspam_scraper(pathname):
	index_skip_subjectline = 2
	list_emails = list()
	files = os.listdir(pathname)
	files.sort()

	for i in range(0, len(files)):
		with open(pathname + "/" + files[i]) as lingspam:
			lines = lingspam.readlines()
			list_emails.append(lines[index_skip_subjectline])

	'''
		once the files are sorted, since the datasets are a 50/50 split of spam and non-spam 
		we can easily label them
	'''
	split_set = int(len(list_emails)/2)
	set_classifications = ([0]*split_set + [1]*split_set)

	return list_emails, set_classifications


def normalize(email_list):
	stop_words = set(stopwords.words('english'))
	punc = set(string.punctuation)
	lemma = WordNetLemmatizer()

	no_stopwords = " ".join([word for word in email_list.lower().split() if word not in stop_words])
	no_punc = ''.join([char for char in no_stopwords if char not in punc])
	normalized_text = " ".join(lemma.lemmatize(word) for word in no_punc.split())

	return normalized_text

def pre_process(email_list):
	normalized_emails = list()

	for i in range(len(email_list)):
		normalized_emails.append(normalize(email_list[i]))
	
	return normalized_emails

def make_dict(email_list):
	all_words = list()

	for i in range(len(email_list)):
		all_words.extend(email_list[i].split())
			
	dictionary = Counter(all_words)
	dictionary = dictionary.most_common(3000)

	return dictionary

def extract_feature_vector(email, dictionary):
	feature_vector = list()
	words = email.split()

	for entry in dictionary:
		feature_vector.append(words.count(entry[0]))

	return feature_vector

def feature_extraction(email_list, dictionary):
	features = list()

	for i in range(len(email_list)):
		features.append(extract_feature_vector(email_list[i], dictionary))

	return features



''' main '''

lingspam_test, lingspam_train, lingspam_test_classification, lingspam_train_classification = load_lingspam_data("spam-non-spam-dataset")
print("\n-------------------- Data has been loaded --------------------\n")

''' Data Pre-Processing '''

# Lingspam
lingspam_train = pre_process(lingspam_train)
lingspam_test = pre_process(lingspam_test)

print("--------------------  Pre-Processing Complete --------------------\n")

''' Dictionary Creation '''

# Lingspam
dictionary = make_dict(lingspam_train)

print("-------------------- Dictionary Creation Complete --------------------\n")

''' Feature Extraction '''

# Lingspam
lingspam_train_vec = feature_extraction(lingspam_train, dictionary)
lingspam_test_vec = feature_extraction(lingspam_test, dictionary)

print("-------------------- Feature Extraction Complete --------------------\n")

''' Machine Learning Models '''

# LinearSVC Classifier
model_svc = svm.LinearSVC(max_iter=5000)
model_svc.fit(lingspam_train_vec, lingspam_train_classification)

# Random Forest Classifier
model_rfc = RandomForestClassifier(n_estimators=2000)
model_rfc.fit(lingspam_train_vec, lingspam_train_classification)

# MLP Classifier
model_mlp_relu = MLPClassifier(hidden_layer_sizes=(5,5,5,5,5,6,6,6,6,6,6,6,3,2,1), activation="relu", max_iter=7000, random_state=1)
model_mlp_relu.fit(lingspam_train_vec, lingspam_train_classification)

print("...Training Complete...\n")

''' Results for models '''

# Linear SVC Results
model_svc_res = model_svc.predict(lingspam_test_vec)

# Random Forest Classifier Results
model_rfc_res = model_rfc.predict(lingspam_test_vec)

# Multilayer Perceptron (relu and logistic) Results
model_mlp_relu_res = model_mlp_relu.predict(lingspam_test_vec)
model_mlp_sig_res = model_mlp_sig.predict(lingspam_test_vec)

print("Results for Machine Learning Models on Lingspam Data\n")
print("-----------------------------------------------\n")
print("Linear Support Vector Machine Classifier Results\n")
print("Accuracy Score: {}".format(metrics.accuracy_score(lingspam_test_classification, model_svc_res)))
print("Precision Score: {}".format(metrics.precision_score(lingspam_test_classification, model_svc_res)))
print("Recall: {}".format(metrics.recall_score(lingspam_test_classification, model_svc_res)))
print("F1 Score: {}".format(metrics.f1_score(lingspam_test_classification, model_svc_res)))
print("-----------------------------------------------\n")
print("Random Forest Classifier Results\n")
print("Accuracy Score: {}".format(metrics.accuracy_score(lingspam_test_classification, model_rfc_res)))
print("Precision Score: {}".format(metrics.precision_score(lingspam_test_classification, model_rfc_res)))
print("Recall: {}".format(metrics.recall_score(lingspam_test_classification, model_rfc_res)))
print("F1 Score: {}".format(metrics.f1_score(lingspam_test_classification, model_rfc_res)))
print("-----------------------------------------------\n")
print("Multilayer Perceptron Classifier (relu) Results\n")
print("Accuracy Score: {}".format(metrics.accuracy_score(lingspam_test_classification, model_mlp_relu_res)))
print("Precision Score: {}".format(metrics.precision_score(lingspam_test_classification, model_mlp_relu_res)))
print("Recall: {}".format(metrics.recall_score(lingspam_test_classification, model_mlp_relu_res)))
print("F1 Score: {}".format(metrics.f1_score(lingspam_test_classification, model_mlp_relu_res)))
print("-----------------------------------------------\n")

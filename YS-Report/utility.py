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


def load_data(init_folder):
	if "enron" in init_folder:
		test_set, training_set, test_set_classification, training_set_classification = load_enron_data(init_folder)
	else:
		test_set, training_set, test_set_classification, training_set_classification = load_lingspam_data(init_folder)

	return test_set, training_set, test_set_classification, training_set_classification

def load_lingspam_data(init_folder):
	folders = [folder.path for folder in os.scandir(init_folder) if os.path.isdir(folder)]

	test_set, test_set_classification = lingspam_scraper(folders[0])
	training_set, training_set_classification = lingspam_scraper(folders[1])

	print("\n...Data has been loaded...\n")

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

def load_enron_data(init_folder):
	folders = [folder.path for folder in os.scandir(init_folder) if os.path.isdir(folder)]

	test_set, test_set_classification = enron_scraper(folders[0])
	training_set, training_set_classification = enron_scraper(folders[1])

	print("\n...Data has been loaded...\n")

	return test_set, training_set, test_set_classification, training_set_classification

def enron_scraper(pathname):
	index_skip_subjectline = 1
	list_emails = list()
	list_ham = list()
	list_spam = list()
	files = os.listdir(pathname)

	for i in range(len(files)):
		if "ham" in files[i]:
			mail_string = ""
			with open(pathname + "/" + files[i], "rb") as enron:
				for line in enron:
					mail_string += line.decode("latin-1")
				list_ham.append(mail_string)
		else:
			mail_string = ""
			with open(pathname + "/" + files[i], "rb") as enron:
				for line in enron:
					mail_string += line.decode("latin-1")
				list_spam.append(mail_string)
	list_emails = (list_ham + list_spam)
	set_classifications = ([0]*len(list_ham) + [1]*len(list_spam))

	return list_emails, set_classifications




def normalize(email):
	stop_words = set(stopwords.words('english'))
	punc = set(string.punctuation)
	lemma = WordNetLemmatizer()

	no_stopwords = " ".join([word for word in email.lower().split() if word not in stop_words])
	no_punc = ''.join([char for char in no_stopwords if char not in punc])
	normalized_text = " ".join(lemma.lemmatize(word) for word in no_punc.split())

	return normalized_text

def pre_process(email_list):
	normalized_emails = list()

	for i in range(len(email_list)):
		normalized_emails.append(normalize(email_list[i]))

	print("Pre-Processing Complete...\n")

	return normalized_emails

def make_dict(email_list):
	all_words = list()

	for i in range(len(email_list)):
		all_words.extend(email_list[i].split())
			
	dictionary = Counter(all_words)
	dictionary = dictionary.most_common(3000)

	print("Dictionary Creation Complete...\n")

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

	print("Feature Extraction Complete...\n")

	return features

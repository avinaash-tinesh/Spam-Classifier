from utility import *

''' Run Machine Learning Models '''

def run_models(dirname):

	''' Load Emails '''
	testing_set, training_set, test_classification, training_classification = load_data(dirname)

	''' Data Pre-Processing '''
	testing_set = (pre_process(testing_set))
	training_set = pre_process((training_set))

	''' Dictionary Creation '''
	dictionary = make_dict(training_set)

	''' Feature Extraction '''
	training_vec = feature_extraction(training_set, dictionary)
	testing_vec = feature_extraction(testing_set, dictionary)

	''' Model Training '''

	# LinearSVC Classifier
	model_svc = svm.LinearSVC(max_iter=5000)
	model_svc.fit(training_vec, training_classification)

	# Random Forest Classifier
	model_rfc = RandomForestClassifier(n_estimators=2000)
	model_rfc.fit(training_vec, training_classification)

	# MLP Classifier
	model_mlp = MLPClassifier(hidden_layer_sizes=(6,6,6,6,6,6,6,5,5,5,5,5,2), solver="lbfgs", alpha=1e-05, random_state=2)
	model_mlp.fit(training_vec, training_classification)

	print("...Training Complete...\n")

	''' Model Testing and Results '''

	# Linear SVC Results
	model_svc_res = model_svc.predict(testing_vec)

	# Random Forest Classifier Results
	model_rfc_res = model_rfc.predict(testing_vec)

	# Multilayer Perceptron Results
	model_mlp_res = model_mlp.predict(testing_vec)


	print("Results for Machine Learning Models on {}\n".format(dirname))
	print("-----------------------------------------------\n")
	print("Linear Support Vector Machine Classifier Results\n")
	print("Accuracy Score: {}".format(metrics.accuracy_score(test_classification, model_svc_res)))
	print("Precision Score: {}".format(metrics.precision_score(test_classification, model_svc_res)))
	print("Recall: {}".format(metrics.recall_score(test_classification, model_svc_res)))
	print("F1 Score: {}".format(metrics.f1_score(test_classification, model_svc_res)))
	print("\n-----------------------------------------------\n")
	print("Random Forest Classifier Results\n")
	print("Accuracy Score: {}".format(metrics.accuracy_score(test_classification, model_rfc_res)))
	print("Precision Score: {}".format(metrics.precision_score(test_classification, model_rfc_res)))
	print("Recall: {}".format(metrics.recall_score(test_classification, model_rfc_res)))
	print("F1 Score: {}".format(metrics.f1_score(test_classification, model_rfc_res)))
	print("\n-----------------------------------------------\n")
	print("Multilayer Perceptron Classifier (relu) Results\n")
	print("Accuracy Score: {}".format(metrics.accuracy_score(test_classification, model_mlp_res)))
	print("Precision Score: {}".format(metrics.precision_score(test_classification, model_mlp_res)))
	print("Recall: {}".format(metrics.recall_score(test_classification, model_mlp_res)))
	print("F1 Score: {}".format(metrics.f1_score(test_classification, model_mlp_res)))
	print("\n-----------------------------------------------\n")

lingspam = "lingspam-dataset"
enron = "enron-dataset"

run_models(enron)

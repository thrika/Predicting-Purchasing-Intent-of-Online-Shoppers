import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report

from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.decomposition import PCA


def input_transform_data():
    # Read the dataset from CVS file
    data = pd.read_csv('online_shoppers_intention.csv')
    # Label encoder for changing non-numeric data to numeric
    # Used for encoding categorical data
    le = LabelEncoder()
    data['Revenue'] = le.fit_transform(data['Revenue'])
    data['Revenue'].value_counts()
    data['Month'] = le.fit_transform(data['Month'])
    data['Month'].value_counts()
    data['VisitorType'] = le.fit_transform(data['VisitorType'])
    data['VisitorType'].value_counts()
    data['Weekend'] = le.fit_transform(data['Weekend'])
    data['Weekend'].value_counts()

    # The formatted data without the labels
    data_sample = data
    data_sample = data_sample.drop(['Revenue'], axis=1)

    # Storing Labels seperately
    data_label = data['Revenue']
    return data_sample, data_label


# The function for Random Forest Classifier to get the accuracies
def RandomForest_function(training_data, testing_data, training_labels, testing_labels):
    # Training the model using random forest
    model = RandomForestClassifier(n_estimators=10).fit(training_data, training_labels)

    # Predict the labels for the testing data
    predicted_label = model.predict(testing_data)

    # Printing the Classification Report for each Fold
    print("Random Forest: ")
    print(classification_report(testing_labels, predicted_label))

    # Get the training and testing score
    training_score = model.score(training_data, training_labels)
    testing_score = model.score(testing_data, testing_labels)

    # Get the True Negative, False Positive, False Negative and False Positive from the confusion matrix
    tn, fp, fn, tp = confusion_matrix(testing_labels, predicted_label).ravel()

    # Calculate the specificity (True Negative rate) and sensitivity (True Positive rate)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    # Calculating the balanced Accuracy
    bal_acc = (specificity + sensitivity) / 2

    return [training_score, testing_score, specificity, sensitivity, bal_acc]


# The function for SVM Classifier to get the accuracies
def SVM_function(training_data, testing_data, training_labels, testing_labels):
    # Training the model using random forest
    model = SVC(gamma='auto').fit(training_data, training_labels)

    # Predict the labels for the testing data
    predicted_label = model.predict(testing_data)

    # Printing the Classification Report for each Fold
    print("SVM: ")
    print(classification_report(testing_labels, predicted_label))

    # Get the training and testing score
    training_score = model.score(training_data, training_labels)
    testing_score = model.score(testing_data, testing_labels)

    # Get the True Negative, False Positive, False Negative and False Positive from the confusion matrix
    tn, fp, fn, tp = confusion_matrix(testing_labels, predicted_label).ravel()

    # Calculate the specificity (True Negative rate) and sensitivity (True Positive rate)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    # Calculating the balanced Accuracy
    bal_acc = (specificity + sensitivity) / 2

    return [training_score, testing_score, specificity, sensitivity, bal_acc]


# The function for Multi Layer Perceptron Classifier to get the accuracies
def mlp_function(training_data, testing_data, training_labels, testing_labels):
    # Training the model using random forest
    model = MLPClassifier(hidden_layer_sizes=(17), max_iter=40000).fit(training_data, training_labels)

    # Predict the labels for the testing data
    predicted_label = model.predict(testing_data)

    # Printing the Classification Report for each Fold
    print("MLP: ")
    print(classification_report(testing_labels, predicted_label))

    # Get the training and testing score
    training_score = model.score(training_data, training_labels)
    testing_score = model.score(testing_data, testing_labels)

    # Get the True Negative, False Positive, False Negative and False Positive from the confusion matrix
    tn, fp, fn, tp = confusion_matrix(testing_labels, predicted_label).ravel()

    # Calculate the specificity (True Negative rate) and sensitivity (True Positive rate)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    # Calculating the balanced Accuracy
    bal_acc = (specificity + sensitivity) / 2

    return [training_score, testing_score, specificity, sensitivity, bal_acc]


# The function for XGBoost Classifier to get the accuracies
def xgboost_function(training_data, testing_data, training_labels, testing_labels):
    # Training the model using random forest
    model = XGBClassifier(hidden_layer_sizes=(17), max_iter=40000).fit(training_data, training_labels)

    # Predict the labels for the testing data
    predicted_label = model.predict(testing_data)

    # Printing the Classification Report for each Fold
    print("XGBoost: ")
    print(classification_report(testing_labels, predicted_label))

    # Get the training and testing score
    training_score = model.score(training_data, training_labels)
    testing_score = model.score(testing_data, testing_labels)

    # Get the True Negative, False Positive, False Negative and False Positive from the confusion matrix
    tn, fp, fn, tp = confusion_matrix(testing_labels, predicted_label).ravel()

    # Calculate the specificity (True Negative rate) and sensitivity (True Positive rate)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    # Calculating the balanced Accuracy
    bal_acc = (specificity + sensitivity) / 2

    return [training_score, testing_score, specificity, sensitivity, bal_acc]


def run_classifier_functions(data_train, data_test, labels_train, labels_test):
    rf = RandomForest_function(data_train, data_test, labels_train, labels_test)
    svm = SVM_function(data_train, data_test, labels_train, labels_test)
    mlp = mlp_function(data_train, data_test, labels_train, labels_test)
    xgb = xgboost_function(data_train, data_test, labels_train, labels_test)

    return rf, svm, mlp, xgb


if __name__ == "__main__":
    # Function to input and format data from CSV File
    data, labels = input_transform_data()

    cv = KFold(n_splits=5)

    normal_randomforest = np.zeros(5)
    normal_svm = np.zeros(5)
    normal_mlp = np.zeros(5)
    normal_xgb = np.zeros(5)

    ada_normal_randomforest = np.zeros(5)
    ada_normal_svm = np.zeros(5)
    ada_normal_mlp = np.zeros(5)
    ada_normal_xgb = np.zeros(5)

    smo_normal_randomforest = np.zeros(5)
    smo_normal_svm = np.zeros(5)
    smo_normal_mlp = np.zeros(5)
    smo_normal_xgb = np.zeros(5)

    pca_randomforest = np.zeros(5)
    pca_svm = np.zeros(5)
    pca_mlp = np.zeros(5)
    pca_xgb = np.zeros(5)

    pca_ada_randomforest = np.zeros(5)
    pca_ada_svm = np.zeros(5)
    pca_ada_mlp = np.zeros(5)
    pca_ada_xgb = np.zeros(5)

    pca_smo_randomforest = np.zeros(5)
    pca_smo_svm = np.zeros(5)
    pca_smo_mlp = np.zeros(5)
    pca_smo_xgb = np.zeros(5)

    for train_index, test_index in cv.split(data):
        data_train, data_test, labels_train, labels_test = data.iloc[train_index], data.iloc[test_index], \
                                                           labels.iloc[train_index], labels.iloc[test_index]

        # Results without any modifications
        rf, svm, mlp, xgb = run_classifier_functions(data_train, data_test,
                                                     labels_train, labels_test)
        normal_randomforest += rf
        normal_svm += svm
        normal_mlp += mlp
        normal_xgb += xgb

        # With Oversampling ADASYN
        oversamplng_ada = ADASYN(random_state=42)
        ada_data_train, ada_labels_train = oversamplng_ada.fit_resample(data_train, labels_train)

        ada_rf, ada_svm, ada_mlp, ada_xgb = run_classifier_functions(ada_data_train, data_test.values,
                                                                     ada_labels_train, labels_test)
        ada_normal_randomforest += ada_rf
        ada_normal_svm += ada_svm
        ada_normal_mlp += ada_mlp
        ada_normal_xgb += ada_xgb

        # With Oversampling SMOTEENN
        oversamplng_smo = SMOTEENN(random_state=42)
        smo_data_train, smo_labels_train = oversamplng_smo.fit_resample(data_train, labels_train)

        smo_rf, smo_svm, smo_mlp, smo_xgb = run_classifier_functions(smo_data_train, data_test.values,
                                                                     smo_labels_train, labels_test)
        smo_normal_randomforest += smo_rf
        smo_normal_svm += smo_svm
        smo_normal_mlp += smo_mlp
        smo_normal_xgb += smo_xgb

        # PCA on normal data
        pca = PCA(0.95).fit(data_train)
        pca_data_train = pca.transform(data_train)
        pca_data_test = pca.transform(data_test)

        pca_rf1, pca_svm1, pca_mlp1, pca_xgb1 = run_classifier_functions(pca_data_train, pca_data_test,
                                                                     labels_train, labels_test)
        pca_randomforest += pca_rf1
        pca_svm += pca_svm1
        pca_mlp += pca_mlp1
        pca_xgb += pca_xgb1

        # PCA on ADASYN oversampled data
        pca = PCA(0.95).fit(ada_data_train)
        pca_ada_data_train = pca.transform(ada_data_train)
        pca_ada_data_test = pca.transform(data_test)

        pca_ada_rf1, pca_ada_svm1, pca_ada_mlp1, pca_ada_xgb1 = run_classifier_functions(pca_ada_data_train,
                                                                                         pca_ada_data_test,
                                                                                         ada_labels_train,
                                                                                         labels_test)
        pca_ada_randomforest += pca_ada_rf1
        pca_ada_svm += pca_ada_svm1
        pca_ada_mlp += pca_ada_mlp1
        pca_ada_xgb += pca_ada_xgb1

        # PCA on SMOTEENN oversampled data
        pca = PCA(0.95).fit(smo_data_train)
        pca_smo_data_train = pca.transform(smo_data_train)
        pca_smo_data_test = pca.transform(data_test)

        pca_smo_rf1, pca_smo_svm1, pca_smo_mlp1, pca_smo_xgb1 = run_classifier_functions(pca_smo_data_train,
                                                                                         pca_smo_data_test,
                                                                                         smo_labels_train,
                                                                                         labels_test)
        pca_smo_randomforest += pca_smo_rf1
        pca_smo_svm += pca_smo_svm1
        pca_smo_mlp += pca_smo_mlp1
        pca_smo_xgb += pca_smo_xgb1

    print("Normal")
    print(normal_randomforest/len(normal_randomforest))
    print(normal_svm / len(normal_svm))
    print(normal_mlp / len(normal_mlp))
    print(normal_xgb / len(normal_xgb))

    print("ADA")
    print(ada_normal_randomforest / len(ada_normal_randomforest))
    print(ada_normal_svm / len(ada_normal_svm))
    print(ada_normal_mlp / len(ada_normal_mlp))
    print(ada_normal_xgb / len(ada_normal_xgb))

    print("SMO")
    print(smo_normal_randomforest / len(smo_normal_randomforest))
    print(smo_normal_svm / len(smo_normal_svm))
    print(smo_normal_mlp / len(smo_normal_mlp))
    print(smo_normal_xgb / len(smo_normal_xgb))

    print("PCA")
    print(pca_randomforest / len(pca_randomforest))
    print(pca_svm / 5)
    print(pca_mlp / 5)
    print(pca_xgb / 5)

    print("PCA + ADASYN")
    print(pca_ada_randomforest / len(pca_ada_randomforest))
    print(pca_ada_svm / len(pca_ada_svm))
    print(pca_ada_mlp / len(pca_ada_mlp))
    print(pca_ada_xgb / len(pca_ada_xgb))

    print("PCA + SMOTEENN")
    print(pca_smo_randomforest / len(pca_smo_randomforest))
    print(pca_smo_svm / len(pca_smo_svm))
    print(pca_smo_mlp / len(pca_smo_mlp))
    print(pca_smo_xgb / len(pca_smo_xgb))

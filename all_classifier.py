import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA
from skfeature.function.information_theoretical_based import LCSI

from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score


def input_transform_data():
    data = pd.read_csv('online_shoppers_intention.csv')

    le = LabelEncoder()
    data['Revenue'] = le.fit_transform(data['Revenue'])
    data['Revenue'].value_counts()
    data['Month'] = le.fit_transform(data['Month'])
    data['Month'].value_counts()
    data['VisitorType'] = le.fit_transform(data['VisitorType'])
    data['VisitorType'].value_counts()
    data['Weekend'] = le.fit_transform(data['Weekend'])
    data['Weekend'].value_counts()

    x = data
    x = x.drop(['Revenue'], axis=1)
    y = data['Revenue']
    return x, y


def RandomForest(x_train, x_test, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    cr = classification_report(y_test, y_pred)
    print(cr)
    trsc = model.score(x_train, y_train)
    print("Training: ", trsc)
    tssc = model.score(x_test, y_test)
    print("Test: ", tssc)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sp = tn / (tn + fp)
    print("Specificity: ", sp)
    sn = tp / (tp + fn)
    print("Sensitivity: ", sn)

    f1 = f1_score(y_test, y_pred, average='macro')
    print("F1 Score: ", f1)

    bal_acc = (sn+sp)/2
    print("Balanced Accuracy: ",bal_acc)
    return [cm, cr, trsc, tssc, sp, sn, f1,bal_acc]


def support_vector_machine(x_train, x_test, y_train, y_test):
    model = SVC(gamma='auto')
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    cr = classification_report(y_test, y_pred)
    print(cr)
    trsc = model.score(x_train, y_train)
    print("Training: ", trsc)
    tssc = model.score(x_test, y_test)
    print("Test: ", tssc)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sp = tn / (tn + fp)
    print("Specificity: ", sp)
    sn = tp / (tp + fn)
    print("Sensitivity: ", sn)

    f1 = f1_score(y_test, y_pred, average='macro')
    print("F1 Score: ", f1)

    bal_acc = (sn + sp) / 2
    print("Balanced Accuracy: ", bal_acc)
    return [cm, cr, trsc, tssc, sp, sn, f1, bal_acc]


def perceptron(x_train, x_test, y_train, y_test):
    model = MLPClassifier(hidden_layer_sizes=(17), max_iter=40000)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    cr = classification_report(y_test, y_pred)
    print(cr)
    trsc = model.score(x_train, y_train)
    print("Training: ", trsc)
    tssc = model.score(x_test, y_test)
    print("Test: ", tssc)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sp = tn / (tn + fp)
    print("Specificity: ", sp)
    sn = tp / (tp + fn)
    print("Sensitivity: ", sn)

    f1 = f1_score(y_test, y_pred, average='macro')
    print("F1 Score: ", f1)

    bal_acc = (sn + sp) / 2
    print("Balanced Accuracy: ", bal_acc)
    return [cm, cr, trsc, tssc, sp, sn, f1, bal_acc]


def xgboost(x_train, x_test, y_train, y_test):
    model = XGBClassifier()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    cr = classification_report(y_test, y_pred)
    print(cr)
    trsc = model.score(x_train, y_train)
    print("Training: ", trsc)
    tssc = model.score(x_test, y_test)
    print("Test: ", tssc)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sp = tn / (tn + fp)
    print("Specificity: ", sp)
    sn = tp / (tp + fn)
    print("Sensitivity: ", sn)

    f1 = f1_score(y_test, y_pred, average='macro')
    print("F1 Score: ", f1)

    bal_acc = (sn + sp) / 2
    print("Balanced Accuracy: ", bal_acc)
    return [cm, cr, trsc, tssc, sp, sn, f1, bal_acc]


def mrmr(X, y, **kwargs):
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        F, J_CMI, MIfy = LCSI.lcsi(X, y, gamma=0, function_name='MRMR', n_selected_features=n_selected_features)
    else:
        F, J_CMI, MIfy = LCSI.lcsi(X, y, gamma=0, function_name='MRMR')
    return F, J_CMI, MIfy


if __name__ == "__main__":
    x, y = input_transform_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    final_data = []
    # Normal
    rf = RandomForest(x_train, x_test, y_train, y_test)
    sv = support_vector_machine(x_train, x_test, y_train, y_test)
    pr = perceptron(x_train, x_test, y_train, y_test)
    xg = xgboost(x_train, x_test, y_train, y_test)

    final_data.append(rf)
    final_data.append(sv)
    final_data.append(pr)
    final_data.append(xg)

    # With Oversampling ADASYN
    ovs = ADASYN(random_state=42)
    new_x_train, new_y_train = ovs.fit_resample(x_train, y_train)

    adaOS_rf = RandomForest(new_x_train, x_test.values, new_y_train, y_test)
    adaOS_sv = support_vector_machine(new_x_train, x_test.values, new_y_train, y_test)
    adaOS_pr = perceptron(new_x_train, x_test.values, new_y_train, y_test)
    adaOS_xg = xgboost(new_x_train, x_test.values, new_y_train, y_test)

    final_data.append(adaOS_rf)
    final_data.append(adaOS_sv)
    final_data.append(adaOS_pr)
    final_data.append(adaOS_xg)

    # With Oversampling SMOTEENN
    ovs = SMOTEENN(random_state=42)
    new_x_train, new_y_train = ovs.fit_resample(x_train, y_train)

    smoOS_rf = RandomForest(new_x_train, x_test.values, new_y_train, y_test)
    smoOS_sv = support_vector_machine(new_x_train, x_test.values, new_y_train, y_test)
    smoOS_pr = perceptron(new_x_train, x_test.values, new_y_train, y_test)
    smoOS_xg = xgboost(new_x_train, x_test.values, new_y_train, y_test)

    final_data.append(smoOS_rf)
    final_data.append(smoOS_sv)
    final_data.append(smoOS_pr)
    final_data.append(smoOS_xg)

    # PCA
    pca = PCA().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(pca, y, test_size=0.3, random_state=0)

    # Normal
    pca_rf = RandomForest(x_train, x_test, y_train, y_test)
    pca_sv = support_vector_machine(x_train, x_test, y_train, y_test)
    pca_pr = perceptron(x_train, x_test, y_train, y_test)
    pca_xg = xgboost(x_train, x_test, y_train, y_test)

    final_data.append(pca_rf)
    final_data.append(pca_sv)
    final_data.append(pca_pr)
    final_data.append(pca_xg)

    # With Oversampling ADASYN
    ovs = ADASYN(random_state=42)
    new_x_train, new_y_train = ovs.fit_resample(x_train, y_train)

    pca_adaOS_rf = RandomForest(new_x_train, x_test, new_y_train, y_test)
    pca_adaOS_sv = support_vector_machine(new_x_train, x_test, new_y_train, y_test)
    pca_adaOS_pr = perceptron(new_x_train, x_test, new_y_train, y_test)
    pca_adaOS_xg = xgboost(new_x_train, x_test, new_y_train, y_test)

    final_data.append(pca_adaOS_rf)
    final_data.append(pca_adaOS_sv)
    final_data.append(pca_adaOS_pr)
    final_data.append(pca_adaOS_xg)

    # With Oversampling SMOTEENN
    ovs = SMOTEENN(random_state=42)
    new_x_train, new_y_train = ovs.fit_resample(x_train, y_train)

    pca_smoOS_rf = RandomForest(new_x_train, x_test, new_y_train, y_test)
    pca_smoOS_sv = support_vector_machine(new_x_train, x_test, new_y_train, y_test)
    pca_smoOS_pr = perceptron(new_x_train, x_test, new_y_train, y_test)
    pca_smoOS_xg = xgboost(new_x_train, x_test, new_y_train, y_test)

    final_data.append(pca_smoOS_rf)
    final_data.append(pca_smoOS_sv)
    final_data.append(pca_smoOS_pr)
    final_data.append(pca_smoOS_xg)

    # MRMR
    F, J, M = mrmr(x.values, y, n_selected_features=14)
    F = F.tolist()
    a = x.iloc[:, F]
    x_train, x_test, y_train, y_test = train_test_split(a, y, test_size=0.3, random_state=0)

    # Normal
    mrmr_rf = RandomForest(x_train, x_test, y_train, y_test)
    mrmr_sv = support_vector_machine(x_train, x_test, y_train, y_test)
    mrmr_pr = perceptron(x_train, x_test, y_train, y_test)
    mrmr_xg = xgboost(x_train, x_test, y_train, y_test)

    final_data.append(mrmr_rf)
    final_data.append(mrmr_sv)
    final_data.append(mrmr_pr)
    final_data.append(mrmr_xg)

    # With Oversampling ADASYN
    ovs = ADASYN(random_state=42)
    new_x_train, new_y_train = ovs.fit_resample(x_train, y_train)

    mrmr_adaOS_rf = RandomForest(new_x_train, x_test, new_y_train, y_test)
    mrmr_adaOS_sv = support_vector_machine(new_x_train, x_test, new_y_train, y_test)
    mrmr_adaOS_pr = perceptron(new_x_train, x_test, new_y_train, y_test)
    mrmr_adaOS_xg = xgboost(new_x_train, x_test.values, new_y_train, y_test)

    final_data.append(mrmr_adaOS_rf)
    final_data.append(mrmr_adaOS_sv)
    final_data.append(mrmr_adaOS_pr)
    final_data.append(mrmr_adaOS_xg)

    # With Oversampling SMOTEENN
    ovs = SMOTEENN(random_state=42)
    new_x_train, new_y_train = ovs.fit_resample(x_train, y_train)

    mrmr_smoOS_rf = RandomForest(new_x_train, x_test, new_y_train, y_test)
    mrmr_smoOS_sv = support_vector_machine(new_x_train, x_test, new_y_train, y_test)
    mrmr_smoOS_pr = perceptron(new_x_train, x_test, new_y_train, y_test)
    mrmr_smoOS_xg = xgboost(new_x_train, x_test.values, new_y_train, y_test)

    final_data.append(mrmr_smoOS_rf)
    final_data.append(mrmr_smoOS_sv)
    final_data.append(mrmr_smoOS_pr)
    final_data.append(mrmr_smoOS_xg)

    np.save("acc.npy", final_data)


import numpy as np
import matplotlib.pyplot as plt


def makegraph_perf_met_forall(a):
    bar_name = ["Random Forest", "SVM", "MLP", "XGBoost"]
    ts = [a[0][3], a[1][3], a[2][3], a[4][3]]
    tpr = [a[0][4], a[1][4], a[2][4], a[4][4]]
    tnr = [a[0][5], a[1][5], a[2][5], a[4][5]]
    fscore = [a[0][6], a[1][6], a[2][6], a[4][6]]
    bacc = [a[0][7], a[1][7], a[2][7], a[4][7]]

    index = np.arange(len(bar_name))
    plt.figure(1)
    plt.bar(index - 0.34, ts, color='r', label="Test", width=0.17)
    plt.xticks(index, bar_name, rotation=60)
    plt.bar(index - 0.17, tpr, color='b', label="TPR", width=0.17)
    plt.xticks(index, bar_name, rotation=60)
    plt.bar(index, tnr, color='g', label="TNR", width=0.17)
    plt.xticks(index, bar_name, rotation=60)
    plt.bar(index + 0.17, fscore, color='y', label="F1 Score", width=0.17)
    plt.xticks(index, bar_name, rotation=60)
    plt.bar(index + 0.34, bacc, color='black', label="Balanced Accuracy", width=0.17)
    plt.xticks(index, bar_name, rotation=60)
    plt.title("All Methods Performance Metrics as Bar Graph")
    plt.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
    plt.show()


def makegraph_perf_met_foreach(a):
    class_names = ["RF", "SVM", "MLP", "XGBoost"]
    index = np.arange(len(class_names))
    plt.figure(2)
    plt.subplot(2, 3, 1)
    ts = [a[0][3], a[1][3], a[2][3], a[3][3]]
    plt.title("Testing Accuracy")
    plt.bar(index, ts, color='r', label="Testing Accuracy")
    plt.xticks(index, class_names)

    plt.subplot(2, 3, 2)
    tpr = [a[0][4], a[1][4], a[2][4], a[3][4]]
    plt.title("True Positive rate")
    plt.bar(index, tpr, color='b', label="True Positive rate")
    plt.xticks(index, class_names)

    plt.subplot(2, 3, 3)
    tnr = [a[0][5], a[1][5], a[2][5], a[3][5]]
    plt.title("True Negative rate")
    plt.bar(index, tnr, color='y', label="True Negative rate")
    plt.xticks(index, class_names)

    plt.subplot(2, 3, 4)
    F1score = [a[0][6], a[1][6], a[2][6], a[3][6]]
    plt.title("F1 Score")
    plt.bar(index, F1score, color='g', label="F1 Score")
    plt.xticks(index, class_names)

    plt.subplot(2, 3, 5)
    F1score = [a[0][7], a[1][7], a[2][7], a[3][7]]
    plt.title("Balanced Accuracy")
    plt.bar(index, F1score, color='black', label="Balanced Accuracy")
    plt.xticks(index, class_names)
    plt.show()


def makegraph_ada_smo_foreach(a):
    plt.figure(3)

    plt.subplot(2, 2, 1)
    rf_name = ["RF", "ADA_RF", "SMO_RF"]
    index = np.arange(len(rf_name))
    rf = [a[0][7], a[4][7], a[8][7]]
    plt.title("Random Forest")
    plt.bar(index, rf, color='r', label="Random Forest")
    plt.xticks(index, rf_name)

    plt.subplot(2, 2, 2)
    svm_name = ["SVM", "ADA_SVM", "SMO_SVM"]
    index = np.arange(len(svm_name))
    svm = [a[1][7], a[5][7], a[9][7]]
    plt.title("SVM")
    plt.bar(index, svm, color='b', label="SVM")
    plt.xticks(index, svm_name)

    plt.subplot(2, 2, 3)
    mlp_name = ["MLP", "ADA_MLP", "SMO_MLP"]
    index = np.arange(len(mlp_name))
    mlp = [a[2][7], a[6][7], a[10][7]]
    plt.title("MLP")
    plt.bar(index, mlp, color='g', label="MLP")
    plt.xticks(index, mlp_name)

    plt.subplot(2, 2, 4)
    xgb_name = ["XGB", "ADA_XGB", "SMO_XGB"]
    index = np.arange(len(xgb_name))
    xgb = [a[3][7], a[7][7], a[11][7]]
    plt.title("XGB")
    plt.bar(index, xgb, color='y', label="XGB")
    plt.xticks(index, xgb_name)

    plt.show()


def makegraph_smo_forall(a):
    plt.figure(4)
    smo_name = ["SMO_RF", "SMO_SVM", "SMO_MLP", "SMO_XGB"]
    index = np.arange(len(smo_name))
    smo = [a[8][7], a[9][7], a[10][7], a[11][7]]
    plt.title("SMOTEENN in all Classifiers")
    plt.bar(index, smo, color='y')
    plt.xticks(index, smo_name)
    plt.show()


def makegraph_pca_mrmr_foreach(a):
    plt.figure(5)

    plt.subplot(2, 2, 1)
    rf_name = ["RF", "PCA_RF", "MRMR_RF"]
    index = np.arange(len(rf_name))
    rf = [a[0][7], a[12][7], a[24][7]]
    plt.title("Random Forest")
    plt.bar(index, rf, color='r')
    plt.xticks(index, rf_name)

    plt.subplot(2, 2, 2)
    svm_name = ["SVM", "PCA_SVM", "MRMR_SVM"]
    index = np.arange(len(svm_name))
    svm = [a[1][7], a[13][7], a[25][7]]
    plt.title("SVM")
    plt.bar(index, svm, color='b')
    plt.xticks(index, svm_name)

    plt.subplot(2, 2, 3)
    mlp_name = ["MLP", "PCA_MLP", "MRMR_MLP"]
    index = np.arange(len(mlp_name))
    mlp = [a[2][7], a[14][7], a[26][7]]
    plt.title("MLP")
    plt.bar(index, mlp, color='g')
    plt.xticks(index, mlp_name)

    plt.subplot(2, 2, 4)
    xgb_name = ["XGB", "PCA_XGB", "MRMR_XGB"]
    index = np.arange(len(xgb_name))
    xgb = [a[3][7], a[15][7], a[27][7]]
    plt.title("XGB")
    plt.bar(index, xgb, color='y')
    plt.xticks(index, xgb_name)

    plt.show()


def  makegraph_pca_mrmr_forall(a):
    plt.figure(6)
    pca_name = ["Random Forest", "SVM", "MLP", "XGBoost"]
    index = np.arange(len(pca_name))

    pca = [a[12][7], a[13][7], a[14][7], a[15][7]]
    mrmr = [a[24][7], a[25][7], a[26][7], a[27][7]]
    plt.bar(index - 0.33, pca, color='y', label="PCA", width=0.33)
    plt.xticks(index, pca_name)

    plt.title("PCA and MRMR in all Classifiers")
    plt.bar(index, mrmr, color='b', label="MRMR", width=0.33)
    plt.xticks(index, pca_name)

    plt.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
    plt.show()


def  makegraph_smo_pca_mrmr_foreach(a):
    plt.figure(7)

    plt.subplot(2, 2, 1)
    rf_name = ["SMO_RF", "PCA_SMO_RF", "MRMR_SMO_RF"]
    index = np.arange(len(rf_name))
    rf = [a[8][7], a[20][7], a[32][7]]
    plt.title("Random Forest")
    plt.bar(index, rf, color='r')
    plt.xticks(index, rf_name)

    plt.subplot(2, 2, 2)
    svm_name = ["SMO_SVM", "PCA_SMO_SVM", "MRMR_SMO_SVM"]
    index = np.arange(len(svm_name))
    svm = [a[9][7], a[21][7], a[33][7]]
    plt.title("SVM")
    plt.bar(index, svm, color='b')
    plt.xticks(index, svm_name)

    plt.subplot(2, 2, 3)
    mlp_name = ["SMO_MLP", "PCA_SMO_MLP", "MRMR_SMO_MLP"]
    index = np.arange(len(mlp_name))
    mlp = [a[10][7], a[22][7], a[34][7]]
    plt.title("MLP")
    plt.bar(index, mlp, color='g')
    plt.xticks(index, mlp_name)

    plt.subplot(2, 2, 4)
    xgb_name = ["SMO_XGB", "PCA_SMO_XGB", "MRMR_SMO_XGB"]
    index = np.arange(len(xgb_name))
    xgb = [a[11][7], a[23][7], a[35][7]]
    plt.title("XGB")
    plt.bar(index, xgb, color='y')
    plt.xticks(index, xgb_name)

    plt.show()


def makegraph_smo_pca_mrmr_forall(a):
    plt.figure(8)
    pca_name = ["SMO_Random Forest", "SMO_SVM", "SMO_MLP", "SMO_XGBoost"]
    index = np.arange(len(pca_name))

    pca = [a[21][7], a[22][7], a[23][7], a[24][7]]
    mrmr = [a[32][7], a[33][7], a[34][7], a[35][7]]
    plt.bar(index - 0.33, pca, color='y', label="PCA", width=0.33)
    plt.xticks(index, pca_name)

    plt.title("PCA and MRMR in all Classifiers with SMOTEENN")
    plt.bar(index, mrmr, color='g', label="MRMR", width=0.33)
    plt.xticks(index, pca_name)

    plt.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
    plt.show()


def makegraph_mrmr_smo_forall(a):
    plt.figure(9)
    pca_name = ["SMO_Random Forest", "SMO_SVM", "SMO_MLP", "SMO_XGBoost"]
    index = np.arange(len(pca_name))

    mrmr = [a[32][7], a[33][7], a[34][7], a[35][7]]
    plt.title("MRMR in all Classifiers with SMOTEENN")
    plt.bar(index, mrmr, color='y', label="MRMR")
    plt.xticks(index, pca_name)
    plt.show()

    print("MRMR + SMO + Random Forest", a[32][7])
    print("MRMR + SMO + SVM", a[33][7])
    print("MRMR + SMO + MLP", a[34][7])
    print("MRMR + SMO + XG Boost", a[35][7])


if __name__ == "__main__":
    # NOTE: THE LEGEND IS NOT SHOWING IN THE IMAGES IN PYCHARM
    a = np.load("acc.npy", allow_pickle=True)

    makegraph_perf_met_forall(a)
    makegraph_perf_met_foreach(a)

    makegraph_ada_smo_foreach(a)
    makegraph_smo_forall(a)

    makegraph_pca_mrmr_foreach(a)
    makegraph_pca_mrmr_forall(a)

    makegraph_smo_pca_mrmr_foreach(a)
    makegraph_smo_pca_mrmr_forall(a)

    makegraph_mrmr_smo_forall(a)






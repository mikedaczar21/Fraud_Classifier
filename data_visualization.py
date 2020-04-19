import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator
import numpy as np
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pylab as pl


current_dir = os.path.abspath(os.path.curdir)
graph_dir = os.path.abspath(os.path.join(current_dir, "Saved_Graph_Figures"))



def plot_3d(model_output, data_points, fig_type, model_type, z_label):

    # plt.ion() # Interactive Mode
    fig = plt.figure(figsize=(12, 8))
    t = fig.suptitle('{} Predictions {}'.format(model_type, fig_type), fontsize=14)
    ax = fig.add_subplot(111, projection='3d')



    xs = list(data_points['x'])
    ys = list(data_points['y'])
    zs = list(data_points['z'])

    # scaled_glove = model_output['CNN_Prob_Fraud'] * 200.000
    # bubble_size = scaled_glove

    plot_points = [(x, y, z) for x, y, z in zip(xs, ys, zs)]


    markers = []
    colors = []
    bubble_size = []
    model_pred_col = "{}_Predictions".format(model_type)
    for index, row in model_output.iterrows():
        # Correct prediction
        if (row['Actual Label'] == row[model_pred_col]):

            # Accepted predicted correctly
            if (row['Actual Label'] == 1):

                markers.append("o")
                colors.append("green")
                bubble_size.append(30)

            # Rejected Predicted correctly
            elif (row['Actual Label'] == 0):
                markers.append("s")
                colors.append("grey")
                bubble_size.append(30)

        # Incorrect Predictions
        elif(row['Actual Label'] != row[model_pred_col]):

            # False Negative (FN) missclassification
            if (row['Actual Label'] == 1) and (row[model_pred_col] == 0):

                markers.append("o")
                colors.append("blue")
                bubble_size.append(40)

            # False Positive (FP) missclassification
            elif (row['Actual Label'] == 0) and (row[model_pred_col] == 1):

                markers.append("s")
                colors.append("red")
                bubble_size.append(40)



    for data, color, mark, size in zip(plot_points, colors, markers, bubble_size):
        x, y, z = data
        ax.scatter(x, y, z, alpha=0.1, c=color, edgecolors='none', s=size, marker= mark)

    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_zlabel(z_label)



    # ax.minorticks_on()
    # ax.grid(axis="x", color="black", alpha=.3, linewidth=1, linestyle=":")
    # ax.grid(axis="y", color="black", alpha=.5, linewidth=.5)


    # ax.set_zlim(0, 100)
    # ax.xaxis.set_major_locator(MultipleLocator(50))
    # ax.xaxis.set_minor_locator(MultipleLocator(10))
    # ax.yaxis.set_major_locator(MultipleLocator(100))
    # ax.yaxis.set_minor_locator(MultipleLocator(50))

    fig_path = os.path.join(graph_dir, "{}_Confidence_{}.png".format(model_type, fig_type))
    plt.savefig(fig_path, transparent=True)




def plot_word_shap(word_shap_dict, words_to_plot, plot_type):



    fig = plt.figure(figsize=(12, 20))
    t = fig.suptitle( plot_type, fontsize=14)
    ax = fig.add_subplot(111)

    y_pos = []
    sorted_words = dict(sorted(words_to_plot.items(), key=lambda v: v[1], reverse=True))
    y_pos = [idx for idx, elem in enumerate(sorted_words)]



    ax.barh(y_pos, list(sorted_words.values()), align='center')
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.tick_params(axis='y', which='major', labelsize=10)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_words.keys())
    ax.invert_yaxis()

    ax.set_xlabel('Word Impact to Model')
    ax.set_ylabel('Words')


    # tl = plt.gca().get_yticklabels()
    # maxsize = max([t.get_window_extent().height for t in tl])
    # m = 0.1 # inch margin
    # s = maxsize/plt.gcf().dpi*len(sorted_words)+2*m
    # margin = m/plt.gcf().get_size_inches()[1]
    #
    # plt.gcf().subplots_adjust(bottom=margin, top=1.-margin)
    # plt.gcf().set_size_inches( plt.gcf().get_size_inches()[0], s)

    plt.show()
    plt.tight_layout()

    fig_path = os.path.join(graph_dir, "{}_{}.png".format('RNN', plot_type))
    fig.savefig(fig_path)



def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Scaling features
    # X = StandardScaler().fit_transform(X)
    feat_ranges = []
    for col in enumerate(X.shape[1]):
        feat_range_min = X.iloc[:, col].min() - 1
        feat_range_max = X.iloc[:, col].max() + 1
        feat_ranges.append(np.arange(feat_range_min, feat_range_max, resolution))

    np.meshgrid(feat_ranges)
    # plot the decision surface
    x1_min, x1_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    x2_min, x2_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    if hasattr(classifier, "decision_function"):
        Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:

        Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    Z = Z.reshape(xx.shape)

    # Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)


    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)





def get_feature_importance(model, features, feature_names, orig_feat, model_type, plot):

    if model_type == "XGBoost":
        # xgboost output
        xgb.plot_importance(model)
        pl.title("XGBoost Plot Importance")
        pl.show()
        print("\n")


    shap.initjs()
    shap_explainer = shap.TreeExplainer(model)
    shap_values = shap_explainer.shap_values(features)

    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript
    if plot == 'force':
        for row in range(200,250):
            shap.force_plot(shap_explainer.expected_value, shap_values[row,:], orig_feat.iloc[row,:], matplotlib=True)

    elif plot == 'decision':
        for row in range(200,250):
            shap.decision_plot(shap_explainer.expected_value, shap_values[row,:], orig_feat.iloc[row,:], link='logit')

    # mean importance
    shap.summary_plot(shap_values, features = features, feature_names= feature_names, plot_type="bar")


    shap.summary_plot(shap_values, features)
    print("\n")

    return shap_values


def get_feature_importance_pred(model, features, model_out, feature_names, orig_feat, model_type, plot):

    if model_type == "XGBoost":
        # xgboost output
        xgb.plot_importance(model)
        pl.title("XGBoost Plot Importance")
        pl.show()
        print("\n")


    shap.initjs()
    shap_explainer = shap.TreeExplainer(model)
    shap_values = shap_explainer.shap_values(features)

    model_pred_col = "{}_Predictions".format(model_type)


    correct_list = []
    incorrect_list = []
    # model_out.set_index(model_out.iloc[:, 0], inplace=True)

    for index in model_out.index:

        current_actual_label = model_out.loc[index, 'Actual Label']
        current_prediction =  model_out.loc[index, model_pred_col]
        # Correct prediction
        if ( current_actual_label == current_prediction ):

            # Accepted predicted correctly
            if ( current_actual_label == 1):
                out_dict = {'Row_Index': index, 'Claim_Num': model_out.loc[index,'Claim Number'], 'Pred_Eval': 'Correct Fraud - True Positive'}
                correct_list.append(out_dict)


            # Rejected Predicted correctly
            elif (current_actual_label == 0):
                out_dict = {'Row_Index': index, 'Claim_Num': model_out.loc[index,'Claim Number'], 'Pred_Eval': 'Correct NonFraud - True Negative'}
                correct_list.append(out_dict)

        # Incorrect Predictions
        elif(current_actual_label != current_prediction):

            # False Negative (FN) missclassification
            if (current_actual_label == 1) and (current_prediction == 0):

                out_dict = {'Row_Index': index, 'Claim_Num': model_out.loc[index,'Claim Number'], 'Pred_Eval': 'Incorrect Fraud - False Negative'}
                incorrect_list.append(out_dict)

            # False Positive (FP) missclassification
            elif (current_actual_label == 0) and (current_prediction == 1):

                out_dict = {'Row_Index': index, 'Claim_Num': model_out.loc[index,'Claim Number'], 'Pred_Eval': 'Incorrect NonFraud - False Positive'}
                incorrect_list.append(out_dict)

    correct_df = pd.DataFrame(correct_list, columns = ['Row_Index', 'Claim_Num', 'Pred_Eval'])
    # correct_df.set_index(correct_df['Row_Index'], inplace=True)
    incorrect_df  = pd.DataFrame(incorrect_list, columns = ['Row_Index', 'Claim_Num', 'Pred_Eval'])
    # incorrect_df.set_index(incorrect_df['Row_Index'], inplace=True)

    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript
    if plot == 'force':
        for row in range(12):
            shap.force_plot(shap_explainer.expected_value, shap_values[row,:], orig_feat.iloc[row,:], matplotlib=True)

    elif plot == 'decision':
        # print("\n Correct Plots \n")
        # for elem in range(len(correct_df)):
        #     current_row = correct_df.loc[elem, 'Row_Index']
        #     current_claim_num = correct_df.loc[elem, 'Claim_Num']
        #     print("\nPrediction Eval: {} \n Claim_Num: {} Current Row: {}".format(correct_df.loc[elem, 'Pred_Eval'], current_claim_num, current_row))
        #     shap.decision_plot(shap_explainer.expected_value, shap_values[current_row,:], orig_feat.loc[current_row,:], link='logit')


        print("\n Incorrect Plots \n")
        for elem in range(len(incorrect_df)):

            current_row = incorrect_df.loc[elem, 'Row_Index']
            current_claim_num = incorrect_df.loc[elem, 'Claim_Num']

            print("\nPrediction Eval: {} \n Claim_Num: {} Current Row: {}".format(incorrect_df.loc[elem, 'Pred_Eval'], current_claim_num, current_row))
            fig = shap.decision_plot(shap_explainer.expected_value, shap_values[current_row,:], orig_feat.loc[current_row,:], link='logit')
            # shap.decision_plot(shap_explainer.expected_value, shap_values[row,:], orig_feat.iloc[row,:], link='logit', show=False)
            fig_path = os.path.join(graph_dir, "Decison_Plots" ,"Incorrect", "DecisionPlot_{}_{}.png".format(incorrect_df.loc[row, 'Pred_Eval'],  incorrect_df.loc[row, 'Claim_Num']))
            plt.savefig(fig_path)

    # mean importance
    shap.summary_plot(shap_values, features = features, feature_names= feature_names, plot_type="bar")


    shap.summary_plot(shap_values, features)
    print("\n")

    return shap_values

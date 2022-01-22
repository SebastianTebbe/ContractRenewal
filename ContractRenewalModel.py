#Import Packages
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score 
from sklearn.model_selection import cross_val_score

#Set direcetories
os.chdir("C:/Users/yate9397/Desktop/Ph.D thesis/Peer Effects and Fuel Consumption/Code/Analysis/Machine Learning")
base_dir = "C:/Users/yate9397/Desktop/Ph.D thesis/Peer Effects and Fuel Consumption/Code/Analysis/Machine Learning"
datafile = "C:/Users/yate9397/Desktop/Ph.D thesis/Peer Effects and Fuel Consumption/Externals/Intermediate/Demographics//WorkplaceNetworkVehicle_ContractLeas_YQ.dta"

#Load data
df = pd.read_stata(datafile)

#Change DateTime
df['YQ'] = df['YQ'].dt.strftime('%Y-%m')
df['D_YQ'] = df['YQ']

#Index: PersonxYQ
df.set_index(['Person_ID', 'YQ'], inplace=True)
df = df.sort_values(['Person_ID', 'YQ'])

#Change Outcome to Dummy (0/1)
df['D_NewPetrol'].values[df['D_NewPetrol'].values > 1] = 1
df['D_NewDiesel'].values[df['D_NewDiesel'].values > 1] = 1
df['D_NewPartElectric'].values[df['D_NewPartElectric'].values > 1] = 1

#Number of Times at Contract Renewal 
df['N_AtContractRenewal'] = df.groupby('Person_ID').cumcount()+1
df = df.sort_values(by=['Person_ID', 'N_AtContractRenewal'])

#Turn Time Fixed effects into Dummy Variables
YQ_Dummy = pd.get_dummies(df['D_YQ'])

#Turn Municipality Fixed effects into Dummy Variables
Municipality_Dummy=pd.get_dummies(df['Municipality'])

#Load Industry Code 
df_Ind = df.filter(like="D_Industry_Type")
df_Ind = df_Ind.fillna(0)

#Keep relevent Data & Clean Data (delete missing observations)
df = df[['D_NewPetrol', 'D_NewDiesel', 'D_NewPartElectric', 'Person_Age', 'Gender', 'Gross_Salary', 'Disp_Income', 'Unemployment_Days', 'D_SelfEmployed', 'D_Retired', 'D_MarriedCohabitant', 'D_Children', 'Y_Education', 'D_PartElectric_T4','Person_Age_Plant', 'Gender_Plant', 'Gross_Salary_Plant', 'Disp_Income_Plant', 'Unemployment_Days_Plant', 'D_SelfEmployed_Plant', 'D_Retired_Plant', 'D_MarriedCohabitant_Plant', 'D_Children_Plant', 'Y_Education_Plant', 'N_IndPlant','D_AltFuel_T4', 'Total_Vehicle_T4', 'VMT_T4', 'Engine_Power_T4', 'Service_Weight_T4', 'Lit_100km_T4','Avg_PlantContractRenewalLeas', 'Total_Vehicle', 'N_IndPlant_1Q', 'Q_PartElectric', 'Q_NewRegCar', 'area', 'Fuel_Price', 'Fuel_Cost', 'D_Education_General','D_Education_Pedagogy', 'D_Education_Humanities', 'D_Education_Social', 'D_Education_Math', 'D_Education_Tech', 'D_Education_Agriculture', 'D_Education_Health', 'D_Education_Services',  'D_Education_Unknown', 'N_AtContractRenewal']].copy()
print(df.isnull().sum())
df["Y_Education"] = df["Y_Education"].fillna(df["Y_Education"].median())
df["Person_Age"] = df["Person_Age"].fillna(df["Person_Age"].median())
df["Disp_Income"] = df["Disp_Income"].fillna(df["Disp_Income"].median())
df["Lit_100km_T4"] = df["Lit_100km_T4"].fillna(df["Lit_100km_T4"].median())
df["Q_PartElectric"] = df["Q_PartElectric"].fillna(df["Q_PartElectric"].median())
df["Q_NewRegCar"] = df["Q_NewRegCar"].fillna(df["Q_NewRegCar"].median())
df["area"] = df["area"].fillna(df["area"].median())
df["Fuel_Price"] = df["Fuel_Price"].fillna(df["Fuel_Price"].median())
df["Fuel_Cost"] = df["Fuel_Cost"].fillna(df["Fuel_Cost"].median())
df = df.dropna()

#Separating the dependent and independent data variables into two data frames
X = df[['Person_Age', 'Gender', 'Gross_Salary', 'Disp_Income', 'Unemployment_Days', 'D_SelfEmployed', 'D_Retired', 'D_MarriedCohabitant', 'D_Children', 'Y_Education', 'D_PartElectric_T4','Person_Age_Plant', 'Gender_Plant', 'Gross_Salary_Plant', 'Disp_Income_Plant', 'Unemployment_Days_Plant', 'D_SelfEmployed_Plant', 'D_Retired_Plant', 'D_MarriedCohabitant_Plant', 'D_Children_Plant', 'Y_Education_Plant', 'N_IndPlant','D_AltFuel_T4', 'Total_Vehicle_T4', 'VMT_T4', 'Engine_Power_T4', 'Service_Weight_T4', 'Lit_100km_T4','Avg_PlantContractRenewalLeas', 'Total_Vehicle', 'N_IndPlant_1Q', 'Q_PartElectric', 'Q_NewRegCar', 'area', 'Fuel_Price', 'Fuel_Cost', 'D_Education_General','D_Education_Pedagogy', 'D_Education_Humanities', 'D_Education_Social', 'D_Education_Math', 'D_Education_Tech', 'D_Education_Agriculture', 'D_Education_Health', 'D_Education_Services',  'D_Education_Unknown', 'N_AtContractRenewal']].copy()
X = pd.merge(X, YQ_Dummy, left_index=True, right_index=True)
X = pd.merge(X, Municipality_Dummy, left_index=True, right_index=True)
X = pd.merge(X, df_Ind, left_index=True, right_index=True)
y = df[['D_NewPetrol', 'D_NewDiesel', 'D_NewPartElectric']].copy()

#Generate Classification of Target Variable
def label_car (row):
   if row['D_NewPetrol'] == 1 :
      return 1
   if row['D_NewDiesel'] == 1 :
      return 2
   if row['D_NewPartElectric'] == 1 :
      return 3
   return 4
y=y.apply (lambda row: label_car(row), axis=1)
y=y.to_frame()
y_arr = label_binarize(y, classes=[1,2,3,4])
n_classes = 4
y = pd.DataFrame(y_arr, index=y.index) 
del y_arr

#Standardize Variables 
from sklearn.preprocessing import StandardScaler
X_col = X
scaler = StandardScaler()
print(scaler.fit(X))
X = pd.DataFrame(scaler.transform(X), index=X.index)
X.columns = X_col.columns

#Split data into test and train data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=1, stratify=y)

#Define Class weights
from sklearn.utils import class_weight
class_weight = {0: y.shape[0]/(y.iloc[:,0].sum()*4),
                1: y.shape[0]/(y.iloc[:,1].sum()*4),
                2: y.shape[0]/(y.iloc[:,2].sum()*4),
                3: y.shape[0]/(y.iloc[:,3].sum()*4)}


######################### LOGISTIC REGRESSION ################################
###One-vs-Rest
# Run LogisticRegression model and fit the model to our training data:
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

#Define model
lr_model = LogisticRegression(class_weight='balanced') #balanced/class_weight for weights
#Define the ovr strategy
ovr = OneVsRestClassifier(lr_model)
#Fit model
ovr.fit(X_train, y_train)

#Make predictions
y_pred = ovr.predict(X_test)
y_pred_prob = ovr.predict_proba(X_test)
y_pred_prob_train = ovr.predict_proba(X_train)

#Generate a confusion matrix
y_test=pd.DataFrame(y_test).to_numpy()
y_train=pd.DataFrame(y_train).to_numpy()
conmat=confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
val = np.mat(conmat)
classnames = ['Petrol', 'Diesel', 'Electric', 'None']
df_cm = pd.DataFrame(val, index=classnames, columns=classnames)
df_cm = df_cm.astype('float') / df_cm.sum(axis=1)[:, np.newaxis]

#Plot Figure
plt.figure()
heatmap = sns.heatmap(df_cm, annot=True, cmap='Blues')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Logistic Regression Model')
plt.savefig('Confusion Matrix/Confusion_Matrix_LogRegression.pdf')
plt.show()
plt.close()

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
fpr_train = dict()
tpr_train = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
    fpr_train[i], tpr_train[i], _ = roc_curve(y_train[:, i], y_pred_prob_train[:, i])

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i],  color='black',  label='Main Sample (Test)')
    plt.plot(fpr_train[i], tpr_train[i],  color='black',  ls='dotted',  label='Training Data')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Logistic Regression ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves/ROC_Curve_LogRegression'+str(i)+'.pdf')
    plt.close()

#Compute AUC 
roc_auc_score(y_test,y_pred_prob)

#Compute AUC using Cross-validation
cv_scores = cross_val_score(lr_model, X, y, cv=5, scoring='roc_auc')
print(cv_scores)

#Feature Importance
coefs = np.abs(ovr.coef_[0])
indices = np.argsort(coefs)[::-1]

plt.figure()
plt.title("Feature importances (Logistic Regression)")
plt.bar(range(15), coefs[indices[:15]],
       color="r", align="center")
plt.xticks(range(15), X.columns[indices[:15]], rotation=45, ha='right')
plt.subplots_adjust(bottom=0.3)

###Save the predicted prob 
y_pred_prob = pd.DataFrame(y_pred_prob, index=X_test.index, columns=['Propensity_Petrol_ContractLeas','Propensity_Diesel_ContractLeas','Propensity_EV_ContractLeas','None'])
y_pred_prob.to_stata('Predicted Probabilities/Predicted_Prob_LogRegression.dta')
    



######################### RANDOM FOREST MODEL ################################
### Build Balanced Random Forest 
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.multioutput import MultiOutputClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

#Define model
bal_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample')

#Defining evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)

#Evaluating model
scores = cross_val_score(bal_rf, X_train, y_train, scoring='roc_auc', cv=10)

#Fitting the model and prediction
bal_rf.fit(X_train, y_train)
y_pred_rf =bal_rf.predict(X_test)

#Generate a confusion matrix
y_test=pd.DataFrame(y_test).to_numpy()
y_train=pd.DataFrame(y_train).to_numpy()
conmat=confusion_matrix(y_test.argmax(axis=1), y_pred_rf.argmax(axis=1))
val = np.mat(conmat)
classnames = ['Petrol', 'Diesel', 'Electric', 'None']
df_cm = pd.DataFrame(val, index=classnames, columns=classnames)
df_cm = df_cm.astype('float') / df_cm.sum(axis=1)[:, np.newaxis]

#Plot Figure
plt.figure()
heatmap = sns.heatmap(df_cm, annot=True, cmap='Blues')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Random Forrest Model')
plt.savefig('Confusion Matrix/Confusion_Matrix_RandomForrest.pdf')
plt.close()


#Feature Importance
coefs = bal_rf.feature_importances_
indices = np.argsort(coefs)[::-1]

plt.figure()
plt.title("Feature importances (Random Forests)")
plt.bar(range(15), coefs[indices[:15]],
       color="r", align="center")
plt.xticks(range(15), X.columns[indices[:15]], rotation=45, ha='right')
plt.subplots_adjust(bottom=0.3)




############################# NEURAL NETWORK #################################
#Import Packages
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.callbacks import History

# fix random seed for reproducibility
np.random.seed(21)

#Stohastic Gradient Descent 
epochs = 50
learning_rate = 0.05
momentum = 0.8
decay_rate = learning_rate / epochs
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

#Define dimensions
input_dim = X_train.shape[1]
num_classes = 4
batch_size = 200

#Build the model
model = Sequential()
model.add(Dense(10, activation=tf.nn.relu, kernel_initializer='uniform', input_dim = input_dim))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(10, kernel_initializer='uniform', activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(num_classes, kernel_initializer='uniform', activation=tf.nn.softmax))

#Compile previous build model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

#Define the learning rate change 
def exp_decay(epoch):
    lrate = learning_rate * np.exp(-decay_rate*epoch)
    return lrate
    
#Learning schedule callback
loss_history = History()
lr_rate = LearningRateScheduler(exp_decay)
callbacks_list = [loss_history, lr_rate]

#Train model 
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=callbacks_list,
                    validation_data=(X_test, y_test), 
                    class_weight='class_weight')

#Predict on new data
y_pred_nn = model.predict(X_test)
y_pred_nn_train = model.predict(X_train)

#Plot train vs test accuracy per epoch
plt.figure
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"])
plt.show()

#Plot loss function per epoch
plt.figure
plt.plot(np.sqrt(history.history['loss']), label='train')
plt.plot(np.sqrt(history.history['val_loss']) ,label='val')
plt.title("Model loss function")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"])
plt.show()

#Evaluate model accuraccy on the test set 
accuracy = model.evaluate(X_test, y_test)[1]

#Generate a confusion matrix
y_test=pd.DataFrame(y_test).to_numpy()
y_train=pd.DataFrame(y_train).to_numpy()
conmat=confusion_matrix(y_test.argmax(axis=1), y_pred_nn.argmax(axis=1))
val = np.mat(conmat)
classnames = ['Petrol', 'Diesel', 'Electric', 'None']
df_cm = pd.DataFrame(val, index=classnames, columns=classnames)
df_cm = df_cm.astype('float') / df_cm.sum(axis=1)[:, np.newaxis]

#Plot Figure
plt.figure()
heatmap = sns.heatmap(df_cm, annot=True, cmap='Blues')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Neural Network Model')
plt.savefig('Confusion Matrix/Confusion_Matrix_NeuralNetwork.pdf')
plt.close()

### Binscatter plots 
from scipy import stats
import matplotlib.pyplot as plt
# Predicted-True Probability
for i in range(n_classes):
    Pred = {"Pred_Prob": y_pred_nn[:,i], "True_Prob": y_test[:,i] }
    Prob = pd.DataFrame(Pred)
    
    # Bin values
    Pred = {"Pred_Prob": y_pred_nn[:,i], "True_Prob": y_test[:,i] }
    Prob = pd.DataFrame(Pred)   
    Prob['quantile'] = pd.qcut(Prob["Pred_Prob"], q=50, duplicates='drop')
    Prob_bin = Prob.groupby(['quantile'], as_index=False).mean()
    Prob_bin['quantile_mean'] = Prob_bin.groupby('quantile')['Pred_Prob'].transform('mean')
    Prob_bin["line"]= Prob_bin["Pred_Prob"]
        
    #Plot bin scatter
    plt.plot(Prob_bin.iloc[:,1], Prob_bin.iloc[:,2], 'o', color="gray", label="Main Sample (Test)")
    plt.plot(Prob_bin.iloc[:,1], Prob_bin.iloc[:,4], linestyle='dashed', color='black', label="45-Line")
    plt.ylabel('True Probability')
    plt.xlabel('Predicted Probability')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=3)
    plt.subplots_adjust(bottom=0.25)
    plt.savefig('Binscatter/Pred_Prob_NeuralNetwork'+str(i)+'.pdf')
    plt.close()
    
    del Pred, Prob
    
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
fpr_train = dict()
tpr_train = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve( y_test[:, i], y_pred_nn[:, i])
    fpr_train[i], tpr_train[i], _ = roc_curve(y_train[:, i], y_pred_nn_train[:, i])

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i],  color='black',  label='Main Sample (Test)')
    plt.plot(fpr_train[i], tpr_train[i],  color='black',  ls='dotted',  label='Training Data')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Logistic Regression ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves/ROC_Curve_LogRegression'+str(i)+'.pdf')
    plt.close()
    
    
#Precision-Recall Curves
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_pred_nn[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_pred_nn[:, i])
    
    display = PrecisionRecallDisplay(
    recall=recall[i],
    precision=precision[i],
    average_precision=average_precision[i],
    estimator_name="Precision",
    )
    
    display.plot(color='black')
    plt.savefig('Binscatter/Precision_Recall'+str(i)+'.pdf')
    plt.close()


# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(
    y_test.ravel(), y_pred_nn.ravel()
)
average_precision["micro"] = average_precision_score(y_test, y_pred_nn, average="micro")

display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
    estimator_name="Precision"
)
display.plot()
_ = display.ax_.set_title("Averaged over all classes")

    
###Save the predicted prob 
y_pred_nn = pd.DataFrame(y_pred_nn, index=X_test.index) #Include index
y_pred_nn.columns = ['Propensity_Petrol_ContractLeas','Propensity_Diesel_ContractLeas','Propensity_EV_ContractLeas','None']
y_pred_nn.to_stata('Predicted_Prob_NeuralNetwork.dta')
    



    
    
    
    
    



  




"""


########################### Hyperparameter Tuning #############################
from sklearn.model_selection import RandomizedSearchCV, KFold
from keras.wrappers.scikit_learn import KerasClassifier
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
fpr_train = dict()
tpr_train = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_nn[:, i])
    fpr_train[i], tpr_train[i], _ = roc_curve(y_train[:, i], y_pred_nn_train[:, i])

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i],  color='black',  label='Main Sample (Test)')
    plt.plot(fpr_train[i], tpr_train[i],  color='black',  ls='dotted',  label='Training Data')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Logistic Regression ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves/ROC_Curve_NeuralNetwork'+str(i)+'.pdf')
    plt.close()
    


#Trying Different Weight Initializations




######################### REGULARIZATION ################################
### Lasso (L1)
#Define model
lr_model_lasso = LogisticRegression(penalty="l1", solver='liblinear')
#Define the ovr strategy
ovr_lasso = OneVsRestClassifier(lr_model_ridge)
#Fit model
ovr_lasso.fit(X_train, y_train)
#Make predictions
y_pred = ovr_lasso.predict(X_test)
y_pred_prob = ovr_lasso.predict_proba(X_test)
y_pred_prob_train = ovr_lasso.predict_proba(X_train)

#Generate a confusion matrix
conmat=confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
val = np.mat(conmat)
classnames = ['Petrol', 'Diesel', 'Electric', 'None']
df_cm = pd.DataFrame(val, index=classnames, columns=classnames)
df_cm = df_cm.astype('float') / df_cm.sum(axis=1)[:, np.newaxis]

#Plot Figure
plt.figure()
heatmap = sns.heatmap(df_cm, annot=True, cmap='Blues')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Lasso Regression Model')
plt.savefig('Confusion Matrix/Confusion_Matrix_Lasso.pdf')
plt.show()
plt.close()


# Find the number of nonzero coefficients (selected features)
coefs = ovr_lasso.coef_
print("Total number of features:", coefs.size)
print("Number of selected features:", np.count_nonzero(coefs))



##################### PRINCIAL COMPONENT ANALYSIS #############################
from sklearn.decomposition import PCA
pca = PCA()

pca.fit(X_train)

print(pca.explained_variance_ratio_.cumsum())



######################### DECISION TREE MODEL ################################
# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X_train, y_train)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))

# Decision Tree
tree_model = DecisionTreeClassifier(criterion="entropy", random_state=21)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

#Generate a confusion matrix
conmat=confusion_matrix(y_test.argmax(axis=1), y_pred_tree.argmax(axis=1))
val = np.mat(conmat)
classnames = ['Petrol', 'Diesel', 'Electric', 'None']
df_cm = pd.DataFrame(val, index=classnames, columns=classnames)
df_cm = df_cm.astype('float') / df_cm.sum(axis=1)[:, np.newaxis]

#Plot Figure
plt.figure()
heatmap = sns.heatmap(df_cm, annot=True, cmap='Blues')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Decision Tree Model')
plt.savefig('Confusion Matrix/Confusion_Matrix_DecisionTree.pdf')
plt.close()




######################### RANDOM FOREST MODEL ################################
# Build Random Forest Feature
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 21)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

#Generate a confusion matrix
conmat=confusion_matrix(y_test.argmax(axis=1), y_pred_rf.argmax(axis=1))
val = np.mat(conmat)
classnames = ['Petrol', 'Diesel', 'Electric', 'None']
df_cm = pd.DataFrame(val, index=classnames, columns=classnames)
df_cm = df_cm.astype('float') / df_cm.sum(axis=1)[:, np.newaxis]

#Plot Figure
plt.figure()
heatmap = sns.heatmap(df_cm, annot=True, cmap='Blues')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Random Forrest Model')
plt.savefig('Confusion Matrix/Confusion_Matrix_RandomForrest.pdf')
plt.close()


#Feature Importance
coefs = rf_model.feature_importances_
indices = np.argsort(coefs)[::-1]

plt.figure()
plt.title("Feature importances (Random Forests)")
plt.bar(range(15), coefs[indices[:15]],
       color="r", align="center")
plt.xticks(range(15), X.columns[indices[:15]], rotation=45, ha='right')
plt.subplots_adjust(bottom=0.3)
plt.show()
"""
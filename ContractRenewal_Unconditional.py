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
datafile = "C:/Users/yate9397/Desktop/Ph.D thesis/Peer Effects and Fuel Consumption/Externals/Intermediate/Demographics/WorkplaceNetworkVehicle_YQ.dta"

## LOGISTIC REGRESSION 
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

#Load data
itr = pd.read_stata(datafile, chunksize=1000000)

#Load data in chunks and predict 
for count, chunk in enumerate(itr):
    print(count)
    df = chunk 
    
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


    ###Save the predicted prob 
    y_pred_prob = pd.DataFrame(y_pred_prob, index=X_test.index, columns=['Propensity_Petrol_ContractLeas','Propensity_Diesel_ContractLeas','Propensity_EV_ContractLeas','None'])
    y_pred_prob.to_stata('Predicted Probabilities/PredictedProb_Uncond_LogRegression'+str(count)+'.dta')







































"""
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

#Define Class weights
from sklearn.utils import class_weight
class_weight = {0: y.shape[0]/(y.iloc[:,0].sum()*4),
                1: y.shape[0]/(y.iloc[:,1].sum()*4),
                2: y.shape[0]/(y.iloc[:,2].sum()*4),
                3: y.shape[0]/(y.iloc[:,3].sum()*4)}

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
                    class_weight=class_weight)

#Predict on new data
y_pred_nn = model.predict(X_test)
y_pred_nn_train = model.predict(X_train)



























################# Uncond. Prob - Neural Network ###############################

datafile = "C:/Users/yate9397/Desktop/Ph.D thesis/Peer Effects and Fuel Consumption/Externals/Intermediate/Demographics/WorkplaceNetworkVehicle_YQ.dta"

#Load data
itr = pd.read_stata(datafile, chunksize=5000000)

#Load data in chunks and predict 
for count, chunk in enumerate(itr):
    print(count)
    df = chunk 
    
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
        
    #Define Class weights
    from sklearn.utils import class_weight
    class_weight = {0: y.shape[0]/(y.iloc[:,0].sum()*4),
                    1: y.shape[0]/(y.iloc[:,1].sum()*4),
                    2: y.shape[0]/(y.iloc[:,2].sum()*4),
                    3: y.shape[0]/(y.iloc[:,3].sum()*4)}
    
    # fix random seed for reproducibility
    np.random.seed(21)
    
    #Stohastic Gradient Descent 
    epochs = 50
    learning_rate = 0.05
    momentum = 0.8
    decay_rate = learning_rate / epochs
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    
    #Define dimensions
    input_dim = X.shape[1]
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
    history = model.fit(X, y,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks_list,
                        validation_data=(X, y), 
                        class_weight=class_weight)

    #Predict on new data
    y_pred_nn = model.predict(X)

    ###Save the predicted prob 
    y_pred_nn = pd.DataFrame(y_pred_nn, index=X.index) #Include index
    y_pred_nn.columns = ['Propensity_Petrol_ContractLeas','Propensity_Diesel_ContractLeas','Propensity_EV_ContractLeas','None']
    y_pred_nn.to_stata('Predicted Probabilities/PredictedProb_Uncond_NeuralNetwork'+str(count)+'.dta')
    

"""
    

    
    



  



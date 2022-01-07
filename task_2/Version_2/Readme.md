In this project, our aim is to predict sepsis in patients. 

We start by addressing the missing features both in training and test set. we impute the missing values of each feature using mean value from first nearest neighbours . This is done for each unique patient id. Sklearn’s KNNImputer class is used for the imputation.
Following the data imputation, we extract seven new standard set of features - minimum, maximum, mean, standard deviation, median, skewness, and kurtosis from each feature for every unique patient id.
These new features for every patient are then used to fit the model. 

For task 1 and 2, RandomForestClassifier gave better auc score than SVC.

For task 3, again RandomForestRegressor was found to give better r2 score than linear regressor except for LABEL_SaO2. 


  

For subtask 1 and 2, we find random forest classifier to give better roc_auc scores on the validation set. Sklearn’s RandomForestClassifier is used for the training. Predictions between 0 and 1 are made by predicting the probability on the test data. 


For subtask 3, we find random forest regressor to give better r2 score on the validation set except LABEL_SpO2 for which linear regression gave better r2 score. Again, we use Sklearn’s relevant classes for training. 

On further hyperparameter tuning of both random forest classifier and random forest regressor, we find n_estimator = 200 to give better roc_auc and r2 score, respectively. 

To obtain the solution of subtasks 1 and 2, we use a random forest classifier. For subtask 3, we use linear regression - to predict mean value of SpO2 and random forest regression to predict mean values of the rest of key vital signs.  

# **Meta data**

'Temp' is the body temperature [Celsius]

'RRate' is the respiration rate of the patient [breath/min]

ABPm, ABPd, ABPs are the mean arterial, diastolic and systolic blood pressures of the patient [mmHg],

'Heartrate' is the number of heart beats per minute [heart beats/min],

'SpO2' is pulse oximetry-measured oxygen saturation of the blood [%].

'EtCo2' is the CO2 pressure during expiration [mmHg]. 

PTT: a test which measures the time it takes for a blood clot to form [sec.]
BUN: Blood urea nitrogen concentration [in mg per dl]
Lactate: Lactate acid concentration [in mg per dl]
Hgb: Haemoglobin concentration [g per dl]
HCO3: Bicarbonate concentration [mmol per l]
BaseExcess: Base excess measured in a blood gas analysis [mmol per l]
Fibrinogen: A protein produced by the liver. This protein helps stop bleeding by helping blood clots to form. Concentration [mg per dl]
Phosphate: Phosphate concetration [mg per dl]
WBC: White blood cell count in blood [number of 1000s per microliter]
Creatinine: Serum creatinine concentration used to determine renal function [mg per dl]
PaCO2: Partial pressure of CO2 in arterial blood [mmHg] indicates effectiveness of lung function
AST: Aspartate transaminase, a clinical test determining liver health [International unit per liter, biological activity]
FiO2: Fraction of inspired oxygen in %
Platelets: Thromocyte count in blood [numbers of 1000s per microliter]
SaO2: Oxygen saturation in arterial blood analyzed with blood gas analysis [%]
Glucose: Concentration of serum glucose [in mg per dl]
Magnesium: Concentration of magnesium in blood [mmol per dl]
Potassium: Concentration of potassium in blood [mmol per liter]
Calcium: Concentration of calcium in blood [mg per dl]
Alkalinephos: Biological activity of the enzyme Alkaline phosphotase [International unit per liter]
Bilirubin_direct: Bilirubin concentration of conjugated bilirubin [mg per dl]
Chloride: Chloride concentration in blood [mmol per l]
Hct: Volume percentage of red blood cells in the blood [%]
Bilirubin_total: Bilirubin concentration including conjugated / unconjugated bilirubin [mg per dl]
TroponinI: Concentration of troponin in the blood [ng per ml]
pH: Measurement of the acidity or alkalinity of the blood, with a standard unit for pH.

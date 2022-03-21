import pandas as pd
import pickle 

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def train_model(filepath:str,label:str) -> str:
    df = pd.read_csv(filepath)
    target = label
    protected_attributes = ["Age"]

    y = df[target]
    X = df.drop([target, "Age"], axis=1)

    ct = ColumnTransformer([("ohe", OneHotEncoder(), X.select_dtypes(include=["object"]).columns.tolist())])

    scaler = StandardScaler(with_mean=False)

    # MODELS = [
    #     {"model_name": "German Credit Risk-SGD", "deployment_name": "German Credit Risk-SGD Deployment", "model": Pipeline([("ct", ct), ("scaler", scaler), ("clf", SGDClassifier(loss="modified_huber"))])},
    #     {"model_name": "German Credit Risk-RF", "deployment_name": "German Credit Risk-RF Deployment", "model": Pipeline([("ct", ct), ("scaler", scaler), ("clf", RandomForestClassifier())])},
    #     {"model_name": "German Credit Risk-SVM", "deployment_name": "German Credit Risk-SVM Deployment", "model": Pipeline([("ct", ct), ("scaler", scaler), ("clf", SVC(probability=True))])},
    # ]

    pipeline = Pipeline([("ct", ct), ("scaler", scaler), ("clf", RandomForestClassifier())])
    pipeline.fit(X,y)

    # Save the Model as a Pickle File

    with open("gcr_pipeline.pkl",'wb') as f:
        pickle.dump(pipeline, f) 


    return pipeline


if __name__== "__main__":
    ## sample input for the model
    
    """
   {"fields": ["CheckingStatus",
  "LoanDuration",
  "CreditHistory",
  "LoanPurpose",
  "LoanAmount",
  "ExistingSavings",
  "EmploymentDuration",
  "InstallmentPercent",
  "Sex",
  "OthersOnLoan",
  "CurrentResidenceDuration",
  "OwnsProperty",
  "InstallmentPlans",
  "Housing",
  "ExistingCreditsCount",
  "Job",
  "Dependents",
  "Telephone",
  "ForeignWorker"],
 "values": [["less_0",
   10,
   "all_credits_paid_back",
   "car_new",
   250,
   "500_to_1000",
   "4_to_7",
   3,
   "male",
   "none",
   2,
   "real_estate",
   "none",
   "rent",
   1,
   "skilled",
   1,
   "none",
   "yes"]]}
    """
    print(train_model(filepath='german_credit_data_biased_training.csv',label='Risk'))


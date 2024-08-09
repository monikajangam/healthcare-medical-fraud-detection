"""
Author: Amila Viraj
Email: amilavir@buffalo.edu
Date: 05-03-2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import pickle

# Load the model and scaler
loaded_model = xgb.XGBClassifier()
loaded_model.load_model('phase3/model/xgb_model.json')
with open('phase3/med_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# function to check if physicians are the same
def check_physicians_same(record):
    '''
    Function to keep track of physicians. check if the attending, operating and other physicians are the same
    0: if all the physicians are the same
    1: if attending and operating physicians are the same but different from the other physician
    2: if attending and other physicians are the same but different from the operating physician
    3: if operating and other physicians are the same but different from the attending physician
    4: if all the physicians are different
    '''
    same_attending_and_operating = record['AttendingPhysician'] == record['OperatingPhysician']
    same_operating_and_other = record['OperatingPhysician'] == record['OtherPhysician']
    same_attending_and_other = record['AttendingPhysician'] == record['OtherPhysician']
    if same_attending_and_operating and same_operating_and_other:
        return 0
    elif same_attending_and_operating and not same_operating_and_other:
        return 1
    elif same_attending_and_other and not same_operating_and_other:
        return 2
    elif same_operating_and_other and not same_attending_and_operating:
        return 3
    else:
        return 4

# Define the preprocessing and prediction function
def preprocess_data(beneficiary_data, inpatient_data, outpatient_data):
    merged_data = pd.merge(inpatient_data, outpatient_data, on=list(set(inpatient_data.columns) & set(outpatient_data.columns)), how='outer')
    # print(merged_data)
    merged_data = pd.merge(merged_data, beneficiary_data, on='BeneID', how='outer')
    # print(merged_data)

    merged_data['ClaimStartDt'] = pd.to_datetime(merged_data['ClaimStartDt'])
    merged_data['ClaimEndDt'] = pd.to_datetime(merged_data['ClaimEndDt'])

    merged_data['DOB'] = pd.to_datetime(merged_data['DOB'])
    merged_data['DOD'] = pd.to_datetime(merged_data['DOD'])

    merged_data['AdmissionDt'] = pd.to_datetime(merged_data['AdmissionDt'])
    merged_data['DischargeDt'] = pd.to_datetime(merged_data['DischargeDt'])

    merged_data['RenalDiseaseIndicator'] = merged_data['RenalDiseaseIndicator'].map({'Y': 1, '0': 0})

    chronic_columns = [col for col in merged_data.columns if 'ChronicCond' in col]

    # change the values of 2 to 0 in the chronic columns
    for col in chronic_columns:
        merged_data[col] = merged_data[col].map({2: 0, 1: 1})

    # adding a flag isDead to indicate if the patient is dead or not
    merged_data['isDead'] = np.where(merged_data['DOD'].isnull(), 0, 1)

    # filling the DOD with the max date if the patient is not dead
    max_date = merged_data['DOD'].max()
    merged_data['DOD'] = merged_data['DOD'].fillna(max_date)

    # for additional features
    merged_data['Age'] = (merged_data['DOD'] - merged_data['DOB']).dt.days // 365
    merged_data['AdmitPeriod'] = (merged_data['DischargeDt'] - merged_data['AdmissionDt']).dt.days
    merged_data['AdmitPeriod'] = merged_data['AdmitPeriod'].fillna(0)
    merged_data['ClaimPeriod'] = (merged_data['ClaimEndDt'] - merged_data['ClaimStartDt']).dt.days
    merged_data['Physician_Same'] = merged_data.apply(check_physicians_same, axis=1)
    merged_data['DiseasesCount'] = merged_data[chronic_columns].sum(axis=1)
    merged_data['PhysiciansCount'] = merged_data[['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']].nunique(axis=1)

    claim_columns = [col for col in merged_data.columns if 'ClmDiagnosisCode' in col]
    merged_data['TotalClaimCodes'] = merged_data[claim_columns].nunique(axis=1)

    claim_proc_columns = [col for col in merged_data.columns if 'ClmProcedureCode' in col]
    merged_data['TotalClaimProcedures'] = merged_data[claim_proc_columns].nunique(axis=1)

    merged_data['DeductibleAmtPaid'] = merged_data['DeductibleAmtPaid'].fillna(0)

    df = merged_data.copy()

    cols_rem = ['AttendingPhysician','OperatingPhysician','OtherPhysician','AdmissionDt',
                'ClmAdmitDiagnosisCode','DischargeDt','DiagnosisGroupCode','ClmDiagnosisCode_1','ClmDiagnosisCode_2',
                'ClmDiagnosisCode_3','ClmDiagnosisCode_4','ClmDiagnosisCode_5','ClmDiagnosisCode_6','ClmDiagnosisCode_7',
                'ClmDiagnosisCode_8','ClmDiagnosisCode_9','ClmDiagnosisCode_10','ClmProcedureCode_1','ClmProcedureCode_2',
                'ClmProcedureCode_3','ClmProcedureCode_4','ClmProcedureCode_5','ClmProcedureCode_6']
    
    # for final output
    orf_df = df.copy()

    df = df.drop(cols_rem, axis=1)
    df = df.drop(['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider', 'DOB', 'DOD'], axis=1)

    scl_cols = ['InscClaimAmtReimbursed',
                'DeductibleAmtPaid',
                'IPAnnualReimbursementAmt',
                'IPAnnualDeductibleAmt',
                'OPAnnualReimbursementAmt',
                'OPAnnualDeductibleAmt',
                'Age',
                'AdmitPeriod',
                'ClaimPeriod',
                'DiseasesCount',
                'PhysiciansCount',
                'TotalClaimCodes',
                'TotalClaimProcedures']
    
    df[scl_cols] = scaler.transform(df[scl_cols])

    # make predictions for new_data using loaded model
    predictions = loaded_model.predict(df)

    # add the predictions to the original dataframe (keep only the columns that are needed: 'BeneID', 'ClaimID', 'Provider', 'PotentialFraud')
    orf_df['PotentialFraud'] = predictions
    orf_df = orf_df[['ClaimID', 'Provider', 'PotentialFraud']]

    return orf_df

# images/healthcare.jpeg

# Streamlit app configuration must be at the top
# st.title(':rainbow[Fraud Detection Model]')
st.markdown("""
<style>
.gradient-title {
    font-size: 42px;
    background: linear-gradient(90deg, rgba(63,94,251,1) 0%, rgba(252,70,107,1) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: bold;
    text-align: center;
}
</style>
<div class='gradient-title'>Healthcare Fraud Detection System</div>
""", unsafe_allow_html=True)

# File uploader for the three CSV files
st.header('Upload Files:')
beneficiary_data = st.file_uploader("#### Upload Beneficiary Data CSV:", type=['csv'])
inpatient_data = st.file_uploader("#### Upload Inpatient Data CSV:", type=['csv'])
outpatient_data = st.file_uploader("#### Upload Outpatient Data CSV:", type=['csv'])

if beneficiary_data and inpatient_data and outpatient_data:
    df_beneficiary = pd.read_csv(beneficiary_data)
    df_inpatient = pd.read_csv(inpatient_data)
    df_outpatient = pd.read_csv(outpatient_data)

    st.write("### Beneficiary Data Preview:")
    st.dataframe(df_beneficiary.head())
    st.write("### Inpatient Data Preview:")
    st.dataframe(df_inpatient.head())
    st.write("### Outpatient Data Preview:")
    st.dataframe(df_outpatient.head())

    if st.button("Predict"):
        with st.spinner('Started Processing!...It will take only a few seconds to get predictions...'):
            output_df = preprocess_data(df_beneficiary, df_inpatient, df_outpatient)
            st.success('Processing complete!')
        st.write("### Prediction Results:")
        st.dataframe(output_df.head())

    if st.button("Reset"):
        st.rerun()

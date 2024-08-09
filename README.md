# Healthcare Fraud Detection

This project focused on developing a solution to detect frauds in healthcare claims.

[![](https://img.shields.io/badge/python-3.11-blue.svg)]()
[![](https://img.shields.io/badge/Model-XGBoost-brightgreen.svg)]()
[![](https://img.shields.io/badge/Made_with-streamlit-important.svg)]()
[![](https://img.shields.io/badge/Course-Data_Intensive_Computing-yellow.svg)]()
[![](https://img.shields.io/badge/Product-Medicare_Fraud_Detection-1f425f.svg)]()

<!-- <br /> -->

## Objectives

    Identify Potential Fraudulent Claims in the Healthcare Industry

## Demo

This is a sample [demo](#) video of the project.

<br />

## ✨ Code-base structure

The project code base structure is as below:

```bash
< PROJECT ROOT >
   |
   |-- phase1/                       # Phase 1 Module
   |    |-- data/                    # Training & Test Data
   |    |-- processed/               # Folder for processed data
   |    |-- pre_processing.ipynb     # notebook file for data pre-processing
   |    |-- phase1_report.pdf        # phase 1 report
   |    
   |    
   |-- phase2/                       # Phase 2 Module
   |    |-- data/                    # Training & Test Data
   |    |-- processed/               # Folder for processed data
   |    |-- modeling_final.ipynb     # notebook file for ML modelling
   |    |-- phase2_report.pdf        # phase 2 report
   |    
   |-- phase3/                       # Phase 3 Module
   |    |-- new_data/                # Folder cobtaining new data
   |    |-- model/                   # Folder containing model
   |           |-- xgb_model.json/   # Final trained model
   |    |-- med_scaler.pkl           # Fitted scaler
   |    |-- streamlit_app.py         # Python script for streamlit web app
   |
   |
   |-- requirements.txt              # Requirements & dependencies
   |
   |-- ************************************************************************
```
<!-- <br /> -->

## Setting up the App Locally

1. Clone the repository:

    ```bash
    $ git clone https://github.com/amilaub/healthcare-medical-fraud-detection.git
    ```

    or Download Zip from <https://github.com/monikajangam/healthcare-medical-fraud-detection.git> & Extract it.

    Then move into the app root folder.


    ```bash
    $ cd healthcare-medical-fraud-detection
    ```

2. Set up new virtual environment (optional):

    ```bash
    $ conda create --name <env-name> python
    $ conda activate <env-name>
    ```

3. Install Requirements: 

    ```bash
    $ pip install -r requirements.txt
    ```

4. To run the app on localhost:

    ```bash
    $ streamlit run phase3/streamlit_app.py
    ```

5. app is running at: 
<http://localhost:8501/>

6. Make Prediction on New Data: 
   
   Follow the Below Steps:
   - Go to the home page
    <img src="images/app 1.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />


   - Upload required csv files which containing the claim data
    <img src="images/app 2.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />
    
        Inside the `new_data` folder under `phase3`, there are 3 csv files for beneficiary data, inpatient data & outpatient data.
        Upload these files in the relevant placeholders.
  

    - Click `Predict` button to get the predictions.
  
        <img src="images/app 3.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

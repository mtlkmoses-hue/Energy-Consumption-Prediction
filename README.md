# Project Title: 
Residential Energy Demand Forecasting (Gaborone, Botswana).

# Project Description: 
This project bridges the gap between Data Science and Utility Management at the BPC. By applying Supervised Machine Learning (Linear Regression), I'm to develop a model that will predicts monthly electricity usage (kWh) based on a property’s Square Footage. By establishing a linear relationship between building size and energy load, this tool assists in transformer load planning and distribution efficiency for utility providers.
Objective: To predict Monthly kWh based on Square Footage using Linear Regression.Dataset: Synthetic Utility Dataset (20 observations).Key Results: Highlight the $R^2$ score and the Correlation (0.998).

## Project Report
A detailed project report is included as `Project1_Report.pdf`.

# Student Information
Name: Kgomotso Moses Motlalekgosi 
Team Members: Individual Project

# Date Completed
March 2026

# The Project Requirements
- Python 3.x
- pandas
- scikit-learn
- matplotlib

# How to run this project
Follow these steps to run the electricity consumption prediction model:

1. Make sure Python 3 is installed on your computer.
2. Make sure the following files are in the same folder:
   - electricity_model.py
   - electricity_data.csv
3. Open Command Prompt (Windows) or Terminal (Mac).
4. Navigate to the folder where the files are saved.
   Example:
   cd Desktop
5. Install required libraries (only the first time):
   pip install pandas scikit-learn matplotlib
6. Run the script:
   python electricity_model.py
7. The program will display the predicted electricity consumption
   for a 2000 square foot house.

   link to the web: https://r5syhphmhi8gw6hzftdg4j.streamlit.app/

# Dataset
The data used in this project is a Synthetic Utility Dataset designed to mirror residential electricity consumption patterns observed in Gaborone and surrounding areas in Botswana. It is modeled after the statistical frameworks used by the Botswana Power Corporation (BPC) and Statistics Botswana to estimate load requirements for new grid connections. Dataset SizeRecords: 20 observations (representing a sample of diverse residential properties).Format: Comma-Separated Values (.csv).Target Type: Continuous Numerical.Features (Attributes)The dataset focuses on the two most critical variables for a simple Linear Regression model:Square_Footage - The total floor area of the residential building (Independent Variable) and Monthly_kWh, The total electricity consumed in a 30-day billing cycle (Target Variable).

# Acknowledgements or Credits
Special thanks to:
- Lecturer, Prof. Ishe Hove for guidance.
- Senior Engineer, Electricity Trading - Mr. Ogaufi Shaun Mokgethi 

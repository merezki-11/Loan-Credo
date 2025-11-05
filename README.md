🏦 Loan Credo



A machine learning–powered app that predicts loan default risk using XGBoost, served through FastAPI and visualized with Streamlit.







Features



\- Cleans and preprocesses raw loan data.

\- Trains an XGBoost classifier to predict loan default likelihood.

\- FastAPI endpoint for real-time predictions.

\- Streamlit interface for user-friendly interaction.



Dataset

The dataset `Loan_Default_List.csv` is **not included** in this repository because of its large size.  
To run the training script (`sample.ipynb`):
1. Place your dataset file in the project root folder.
2. Make sure it’s named **Loan_Default_List.csv**.


How It Works



1\. Training (`sample.py`)



&nbsp;  - Loads and cleans the dataset.

&nbsp;  - Handles missing values and encodes categorical variables.

&nbsp;  - Trains an XGBoost model and saves it with preprocessing assets.



2\. API (`fast\_api.py`)



&nbsp;  - Loads the trained model and metadata.

&nbsp;  - Provides `/predict` and `/health` endpoints for inference.





3\. Frontend (`streamlit\_app.py`)



&nbsp;  - Collects user input.

&nbsp;  - Sends it to the FastAPI backend.

&nbsp;  - Displays prediction and risk level.







Setup and Run



1\. Install dependencies:



&nbsp;  ```bash

&nbsp;  pip install -r requirements.txt



2\. Run the API:



&nbsp;  uvicorn fast\_api:app --reload



3\. Run Streamlit:



&nbsp;   streamlit run streamlit\_app.py



Author

Macnelson – AI/ML Engineer
“Building smarter financial tools with machine learning.”


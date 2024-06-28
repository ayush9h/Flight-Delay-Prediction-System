# Flight Delay Prediction System - ZenML Update

A system made to predict flight delays using Machine Learning models.


## Pre-requisite

Install the NYC Flights Dataset from Kaggle.<br>
[Kaggle Dataset Link](https://www.kaggle.com/varunmarvah/nyc-flights-dataset-exploratory-analysis)


## Installation 

1. Create a virtual environment and activate it
  ```bash
  python -m venv myenv
  cd myenv/Scripts
  activate
```

2. Then install necessary external libraries mentioned below in terminal.

```bash
  pip install streamlit
  pip install joblib
  pip install zenml
  pip install "zenml[server]"
```
or 
```bash
pip install -r requirements.txt

```

3. You can run the notebook from the notebooks directory or run ```python main.py```. 

4. To start the zenml dashboard locally on windows, write  ```zenml up --blocking```for Windows and  ```zenml up``` for Mac, and enter the Zenml Dashboard with username "default".


## Features of the system

- Flight departure delay prediction
- Fast and effective
- User friendly UI using Streamlit


## Screenshots
![Demo](https://github.com/ayush9h/Flight-Delay-Prediction-System/blob/main/notebooks/Screenshot%202024-06-28%20121823.png)

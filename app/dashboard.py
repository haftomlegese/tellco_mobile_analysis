import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import psycopg2
import datetime
import logging

# Configure logging
logging.basicConfig(filename='model_deployment.log', level=logging.INFO)

# Load the telecom data
@st.cache_data
def load_data():
    # Define the connection parameters
    host = 'localhost'
    database = 'tellcodb'
    user = 'postgres'
    password = '1234'

    try:
        # Establish a connection to the database
        connection = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        cursor = connection.cursor()

        # Fetch the telecom dataset
        query = "SELECT * FROM xdr_data;"
        cursor.execute(query)
        telecom_data = cursor.fetchall()

        # Convert fetched data into a pandas DataFrame
        telecom_df = pd.DataFrame(telecom_data, columns=[desc[0] for desc in cursor.description])

        # Close the cursor and connection
        cursor.close()
        connection.close()

        return telecom_df

    except (Exception, psycopg2.Error) as error:
        print("Error while fetching data from PostgreSQL:", error)

# Define functions for model training and prediction
def train_model(data):
    # Compute engagement score
    data['engagement_score'] = data['Avg Bearer TP DL (kbps)'] + data['Avg Bearer TP UL (kbps)']

    # Compute experience score
    data['experience_score'] = data['Avg RTT DL (ms)'] + data['Avg RTT UL (ms)']

    # Calculate satisfaction score as the average of engagement and experience scores
    data['satisfaction_score'] = (data['engagement_score'] + data['experience_score']) / 2

    # Drop rows with missing values
    data.dropna(inplace=True)

    # Split data into features (X) and target (y)
    X = data[['engagement_score', 'experience_score']]
    y = data['satisfaction_score']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Linear Regression model
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    # Predict satisfaction scores on the testing data
    y_pred = regression_model.predict(X_test)

    # Compute Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)

    return regression_model, mse

# Define function for K-means clustering and cluster aggregation
def cluster_analysis(data):
    # Combine engagement, experience, and satisfaction scores into a single DataFrame
    scores_df = data[['engagement_score', 'experience_score', 'satisfaction_score']]

    # Perform K-means clustering on the combined scores
    kmeans = KMeans(n_clusters=2, random_state=42)
    scores_df['cluster'] = kmeans.fit_predict(scores_df[['engagement_score', 'experience_score']])

    # Aggregate the average satisfaction and experience scores per cluster
    cluster_scores = scores_df.groupby('cluster').agg({
        'satisfaction_score': 'mean',
        'experience_score': 'mean'
    }).reset_index()

    return cluster_scores

# Set up Streamlit UI components
st.title('Telecom Data Analysis and Model Monitoring')

# Record code version
code_version = 'v1.0'
logging.info(f'Code Version: {code_version}')

# Record start time
start_time = datetime.datetime.now()
logging.info(f'Start Time: {start_time}')

# Load telecom data
telecom_df = load_data()

# Display the loaded data
st.subheader('Telecom Data')
st.write(telecom_df.head())

# Train the model and display metrics
st.subheader('Model Training and Evaluation')
with st.spinner('Training the model...'):
    model, mse = train_model(telecom_df)
st.success(f'Model training completed. Mean Squared Error: {mse}')

# Display the model parameters
st.write('Model Parameters:')
st.write(model.coef_)
st.write(model.intercept_)
logging.info('Model parameters:')
logging.info(f'Coefficients: {model.coef_}')
logging.info(f'Intercept: {model.intercept_}')

# Perform cluster analysis
st.subheader('Cluster Analysis')
cluster_scores = cluster_analysis(telecom_df)
st.write('Cluster Scores:')
st.write(cluster_scores)

# Record end time
end_time = datetime.datetime.now()
logging.info(f'End Time: {end_time}')

# Calculate total execution time
execution_time = end_time - start_time
logging.info(f'Total Execution Time: {execution_time}')

# Display model tracking information
st.subheader('Model Tracking Report')
st.write(f'Code Version: {code_version}')
st.write(f'Start Time: {start_time}')
st.write(f'End Time: {end_time}')
st.write(f'Total Execution Time: {execution_time}')

# Close the logging file
logging.shutdown()
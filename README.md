# Library-Book-Recommendation-System
Aim: A system that recommends books based on past preferences can improve user satisfaction and engagement. It enhances the reading experience by offering personalized options.

This project is a web-based book recommendation system built with Streamlit and Python. It utilizes a content-based filtering approach to suggest new books to users based on their past reading history, taking into account factors like genre and author. The application is designed to be interactive, allowing users to upload their own datasets and filter recommendations by genre and publication year.

## Features

* **File Upload:** Users can upload their own CSV datasets containing library book data.

* **Data Cleaning:** The application automatically handles messy data, including missing values and inconsistent text.

* **Content-Based Filtering:** The core recommendation engine uses TF-IDF and cosine similarity to find similar books.

* **Interactive Filters:** Users can refine recommendations by selecting specific genres and a range of publication years.

* **Data Visualization:** A bar chart provides a quick visual summary of the most popular books in the dataset.

* **Clean UI:** A user-friendly interface with a sidebar for filters and a main content area for results.

## Project Setup and Requirements

**1. File Structure**

Your project folder should have the following structure:

     /your_project_folder/
     ├── app.py
     ├── library_books_dataset_10000_messy.csv
     └── requirements.txt

* app.py: The main Python script for the Streamlit application.

* library_books_dataset_10000_messy.csv: The sample dataset for the recommendation system.

* requirements.txt: A list of all Python libraries required to run the project.


**2. Virtual Environment**

It is highly recommended to use a virtual environment to manage dependencies.

**Create and activate the environment:**

* Windows:

       python -m venv .venv
      .venv\Scripts\activate

* macOS/Linux:

      python3 -m venv .venv
      source .venv/bin/activate

**3. Requirements**

The project requires the following Python libraries. You can install them by running:

     pip install -r requirements.txt

The contents of requirements.txt should be:

       streamlit
       pandas
       scikit-learn
       matplotlib

## How to Run the Application

Once you have set up the project and installed the dependencies, you can launch the app from your terminal.

1. Open your terminal or command prompt.

2. Navigate to your project directory.

3. Ensure your virtual environment is activated.

Run the following command:

      streamlit run app.py

This will automatically open a new tab in your web browser with the application running on http://localhost:8501

## How to Use the App

* Upload Data: On the web page, click the "Browse files" button and select the library_books_dataset_10000_messy.csv file. The app will automatically process and clean the data.

* View Visualization: A bar chart of the top 5 most borrowed books will appear in the main content area.

* Select User and Filters: Use the sidebar to choose a UserID from the dropdown and apply optional filters for genre and publication year.

* Get Recommendations: Click the "Get Recommendations" button to see a personalized list of book suggestions based on your selections.

## Demo

       https://library-book-recommendation-system-nxdjgylds8qnx6fmy854ud.streamlit.app/

After opening this link you have to upload the data whicb is given in this repo. 

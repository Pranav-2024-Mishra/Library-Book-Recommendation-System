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


# RAG Evaluation Pipeline

This project provides a web-based interface for evaluating the performance of a RAG (Retrieval-Augmented Generation) agent. It uses DeepEval for generating test cases and evaluating metrics, and a Flask backend to orchestrate the process.

## Features

- **Web Interface:** A simple, minimalist frontend to run evaluations and view results.
- **Two Evaluation Modes:** 
    - **Quick Eval:** Runs a smaller set of test cases for a quick check.
    - **Deep Eval:** Runs a more comprehensive set of test cases for a thorough evaluation.
- **Automated Workflow:** The entire pipeline, from test case generation to evaluation, is automated.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  **Start the Flask application:**
    ```bash
    python app.py
    ```

2.  **Open your web browser** and navigate to `http://127.0.0.1:5000`.

3.  **Click the "Run Quick Evaluation" or "Run Deep Evaluation" button** to start the evaluation process.

4.  **View the results** in the web interface.

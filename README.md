# Local RAG Evaluation Pipeline

This project provides a comprehensive, modular pipeline for evaluating Retrieval-Augmented Generation (RAG) systems using local LLMs powered by Ollama and `sentence-transformers`. It automatically generates test cases, runs them against a RAG agent, evaluates the results using `deepeval` metrics, and provides an AI-powered summary for improvements.

The pipeline features two modes:

  * **Quick Eval**: A fast, shallow evaluation with fewer test cases.
  * **Deep Eval**: A more thorough evaluation with a larger and more complex set of test cases.

-----

## ⚙️ Workflow Diagram

The pipeline follows a sequential process orchestrated by the main `run_evaluation.py` script.

```
[ User ]
   |
   V
[ run_evaluation.py ] --(Asks "Quick" or "Deep"?)
   |
   |--(1. Generate Goldens - if needed)--> [ synthesizer_quick.py ] or [ synthesizer_deep.py ]
   |                                          |
   |                                          V
   |                                      (Creates goldens_quick.json or goldens_deep.json)
   |
   |--(2. Run RAG Agent - if needed)-----> [ rag_agent.py ]
   |                                          |
   |                                          V
   |                                      (Creates ..._with_output.json)
   |
   |--(3. Evaluate Metrics in Parallel)--> [ evaluate_metric.py ] (Called 5 times)
   |                                          |
   |                                          V
   |                                      (Creates metric_report.json for each metric)
   |
   V
[ Final Report & AI Summary ] --(Printed to Console & Saved to .txt file)
```

-----

## 📂 File Structure

```
/deep_&_quick/
├── config.py
├── custom_models.py
├── synthesizer_quick.py
├── synthesizer_deep.py
├── rag_agent.py
├── evaluate_metric.py
├── run_evaluation.py
├── SBI_car_policy-1.pdf
├── (Generated Files)...
│   ├── goldens_quick.json
│   ├── goldens_quick_with_output.json
│   ├── answer_relevancy_quick_report.json
│   └── evaluation_report_quick_... .txt
└── ...
```

-----

## 🚀 Setup and Installation

1.  **Install Ollama**: Download and run the [Ollama application](https://ollama.com/) on your machine.
2.  **Pull Models**: Open your terminal and pull the required generation and embedding models.
    ```bash
    ollama pull llama3
    ollama pull nomic-embed-text
    ```
3.  **Python Environment**: Ensure you are in your `voicerax` virtual environment.
4.  **Install Dependencies**: Install all necessary Python packages.
    ```bash
    pip install deepeval ollama sentence-transformers torch langchain langchain-community langchain-ollama faiss-cpu pypdf
    ```

-----

## ▶️ How to Run

Make sure your Ollama application is running in the background. To start the entire pipeline, simply run the main orchestrator script:

```bash
python run_evaluation.py
```

The script will then prompt you to choose between a "Quick" or "Deep" evaluation.

-----

## Detailed File Descriptions

### `run_evaluation.py`

  * **Purpose**: This is the main orchestrator and the **only script you need to run directly**. It manages the entire pipeline from start to finish.
  * **How it's Called**: By the user from the command line (`python run_evaluation.py`).
  * **What it Calls/Uses**:
      * `config.py`: Reads all configurations.
      * `subprocess`: To run the `synthesizer` and `evaluate_metric` scripts in separate processes.
      * `rag_agent.py`: Calls the `generate_rag_responses` function to get the RAG agent's output.
      * `custom_models.py`: Imports `CustomOllamaLLM` to generate the final improvement summary.
  * **Key Functions**:
      * `main()`: Asks the user for input, checks if files exist, and calls the other scripts and functions in the correct order.
      * `run_metric_evaluation()`: A helper function that runs a single metric evaluation in a subprocess and captures its output. This is what enables parallel execution.
      * `summarize_failures()`: Collects all failing test cases, creates a prompt, and calls the `CustomOllamaLLM` to generate actionable improvement suggestions.

### `config.py`

  * **Purpose**: A centralized file to hold all user-configurable settings. This makes it easy to change models, file paths, and parameters without touching the logic scripts.
  * **How it's Called**: It is imported by almost every other Python script in the project.
  * **What it Calls/Uses**: Nothing. It only contains variable definitions.

### `custom_models.py`
*This is the corrected version of custom_models.py from the main repository. Next time please test and commit. T_T*
  * **Purpose**: Defines the custom wrapper classes that allow `deepeval` to interface with local models. This file makes the entire local-first approach possible.
  * **How it's Called**: Imported by `synthesizer_quick.py`, `synthesizer_deep.py`, `evaluate_metric.py`, and `run_evaluation.py`.
  * **What it Calls/Uses**:
      * `sentence_transformers`: To run embeddings directly in Python.
      * `ollama`: To make API calls to the local Ollama server for text generation.
      * `deepeval.models`: To inherit from the required base classes.
  * **Key Classes**:
      * `CustomEmbeddingModel`: A wrapper for `sentence-transformers`. It handles downloading the model on the first run and provides the embedding methods `deepeval` requires.
      * `CustomOllamaLLM`: A wrapper for the `ollama` client. It enhances prompts to ensure reliable JSON output and handles communication with the Ollama server for all generative tasks.

### `synthesizer_quick.py` & `synthesizer_deep.py`

  * **Purpose**: These scripts are responsible for generating the test cases ("goldens") from the source document. The `_quick` version creates a small, simple set, while the `_deep` version creates a larger, more complex set.
  * **How they are Called**: By `run_evaluation.py` using a `subprocess` call if the required golden file does not already exist.
  * **What they Call/Use**:
      * `config.py`: To get model names, file paths, and synthesizer parameters (`QUICK_SYNTH_PARAMS` or `DEEP_SYNTH_PARAMS`).
      * `custom_models.py`: To instantiate the embedding and generation models.
      * `deepeval.synthesizer.Synthesizer`: The core `deepeval` class that performs the test case generation.

### `rag_agent.py`

  * **Purpose**: This script simulates your actual RAG application. It loads the generated goldens, runs each question (`input`) through a LangChain RAG pipeline, and saves the agent's response (`actual_output`) and the documents it retrieved (`retrieval_context`).
  * **How it's Called**: The `generate_rag_responses` function is called by `run_evaluation.py` if the file with RAG outputs doesn't exist.
  * **What it Calls/Uses**:
      * `config.py`: To get model names and the source document path.
      * `langchain`, `langchain_community`, `langchain_ollama`: The core libraries used to build the RAG chain.
  * **Key Functions**:
      * `create_rag_chain()`: Builds the complete LangChain pipeline (PDF loader, text splitter, vector store, retriever, and LLM chain).
      * `generate_rag_responses()`: The main function that orchestrates loading goldens, running them through the RAG chain, and saving the enriched results.

### `evaluate_metric.py`

  * **Purpose**: A reusable, command-line-driven script that calculates a single `deepeval` metric for a given dataset. Its modular design allows `run_evaluation.py` to execute multiple evaluations in parallel.
  * **How it's Called**: By `run_evaluation.py` using a `subprocess` call, once for each metric listed in `config.py`. It receives arguments like the metric name and file paths from the command line.
  * **What it Calls/Uses**:
      * `config.py`: To get the `DEEPEVAL_GENERATION_MODEL` name.
      * `custom_models.py`: To instantiate the `CustomOllamaLLM` to act as the "judge" for the metric.
      * `deepeval.metrics`: To import and use the specific metric class (e.g., `FaithfulnessMetric`).
  * **Key Functions**:
      * `run_evaluation()`: The main logic that loads the data, loops through each test case, calls `metric.measure()`, and saves the results. At the end, it prints the crucial "Average Score" to the console, which is captured by `run_evaluation.py`.

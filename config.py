# config.py

# --- Model Configuration ---
# For DeepEval's Synthesizer and Judge
DEEPEVAL_GENERATION_MODEL = "llama3.2"
DEEPEVAL_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# For your LangChain RAG Agent
LC_LLM_MODEL = "llama3.2"
LC_EMBEDDING_MODEL = "nomic-embed-text"

# --- File Path Configuration ---
SOURCE_DOCUMENT = 'C:/Voicerax/Voicerax/SBI_car_policy-1.pdf'
QUICK_GOLDENS_FILE = 'goldens_quick.json'
DEEP_GOLDENS_FILE = 'goldens_deep.json'
QUICK_GOLDENS_WITH_OUTPUT_FILE = 'goldens_quick_with_output.json' 
DEEP_GOLDENS_WITH_OUTPUT_FILE = 'goldens_deep_with_output.json'   

# --- Synthesizer Configuration ---
QUICK_SYNTH_PARAMS = {"max_goldens_per_context": 1, "num_evolutions": 3}
DEEP_SYNTH_PARAMS = {"max_goldens_per_context": 3, "num_evolutions": 5}

# --- Evaluation Configuration ---
METRICS_TO_RUN = [
    "answer_relevancy", "faithfulness", "contextual_precision",
    "contextual_recall", "contextual_relevancy"
]
PASS_THRESHOLD = 0.8
IMPROVEMENT_THRESHOLD = 0.5
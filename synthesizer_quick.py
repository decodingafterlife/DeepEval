# synthesizer_quick.py

from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig, EvolutionConfig
from deepeval.dataset import EvaluationDataset
from custom_models import CustomEmbeddingModel, CustomOllamaLLM
import config

print("--- Starting Quick Golden Generation ---")

# ... (code to instantiate models and synthesizer is unchanged) ...
embedding_model = CustomEmbeddingModel(model_name=config.DEEPEVAL_EMBEDDING_MODEL)
llm_model = CustomOllamaLLM(model=config.DEEPEVAL_GENERATION_MODEL)
evolution_config = EvolutionConfig(num_evolutions=config.QUICK_SYNTH_PARAMS["num_evolutions"])
synthesizer = Synthesizer(model=llm_model, evolution_config=evolution_config)
context_config = ContextConstructionConfig(embedder=embedding_model, critic_model=llm_model)

goldens = synthesizer.generate_goldens_from_docs(
    document_paths=[config.SOURCE_DOCUMENT],
    context_construction_config=context_config,
    max_goldens_per_context=config.QUICK_SYNTH_PARAMS["max_goldens_per_context"]
)

dataset = EvaluationDataset(goldens=goldens)

# CORRECTED: Save as a single file in the current directory
dataset.save_as(
    file_type="json", 
    directory=".", 
    file_name=config.QUICK_GOLDENS_FILE.replace('.json', '')
)

print(f"\n✅ Quick generation complete. {len(goldens)} goldens saved to '{config.QUICK_GOLDENS_FILE}'")
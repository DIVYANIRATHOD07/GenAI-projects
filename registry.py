"""
Experiment Registry and Integration Layer for GenAI Practical Lab
==================================================================
Maps all 10 practical experiments to metadata, execution handlers,
and validation logic while keeping original files untouched.
"""

import os
import sys
import subprocess
from typing import Dict, Any, List, Optional, Callable

WORKSPACE_DIR = os.path.abspath(os.path.dirname(__file__))

EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "exp1": {
        "id": "exp1",
        "num": 1,
        "title": "Exploratory Data Analysis & Preprocessing on Tabular Datasets",
        "filename": "65-GEN-AI-EX1.ipynb",
        "category": "Data Engineering & Preprocessing",
        "aim": "To load, inspect, visualize, and preprocess standard benchmark tabular datasets using Pandas, Seaborn, and Scikit-Learn.",
        "description": (
            "Performs data science workflows on Titanic, Iris, Tips, and California Housing datasets: "
            "inspects data schemas, missing values, statistical distributions, correlation heatmaps, "
            "and applies scaling (Standard, MinMax, Robust) and categorical encoding."
        ),
        "dependencies": ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "plotly"],
        "has_terminal_input": False,
        "subprocess_safe": True,
        "required_models": [],
        "required_datasets": ["titanic", "iris", "tips", "penguins", "california_housing"],
        "required_keys": [],
        "execution_type": "in_process_and_subprocess",
    },
    "exp2": {
        "id": "exp2",
        "num": 2,
        "title": "Python Operators & TensorFlow Tensor Operations",
        "filename": "EX2_GAI_65.ipynb",
        "category": "Deep Learning Fundamentals",
        "aim": "To evaluate fundamental Python arithmetic/logical/comparison operators and implement multidimensional TensorFlow tensor operations.",
        "description": (
            "Executes arithmetic, assignment, logical, bitwise, and comparison operations with user-supplied integers. "
            "Then builds 0-D to 3-D TensorFlow constants, tests tensor reduction, element-wise math, reshaping, and NumPy conversions."
        ),
        "dependencies": ["tensorflow", "numpy"],
        "has_terminal_input": True,
        "subprocess_safe": False,
        "required_models": [],
        "required_datasets": [],
        "required_keys": [],
        "execution_type": "adapted_interactive_and_piped_subprocess",
    },
    "exp3": {
        "id": "exp3",
        "num": 3,
        "title": "AI Question-Answering System with FLAN-T5",
        "filename": "65-EX3-GEN-AI.ipynb",
        "category": "Natural Language Processing",
        "aim": "To build an instruction-conditioned Question-Answering system using the pre-trained google/flan-t5-base Seq2Seq model.",
        "description": (
            "Loads google/flan-t5-base from Hugging Face Hub, tokenizes instruction prompts with beam search, "
            "and generates grounded natural language answers to user queries."
        ),
        "dependencies": ["torch", "transformers", "accelerate", "sentencepiece", "gradio"],
        "has_terminal_input": True,
        "subprocess_safe": False,
        "required_models": ["google/flan-t5-base (~990MB)"],
        "required_datasets": [],
        "required_keys": [],
        "execution_type": "adapted_model_inference",
    },
    "exp4": {
        "id": "exp4",
        "num": 4,
        "title": "Text-to-Image Generation with Stable Diffusion v1-5",
        "filename": "GENAI-EX4-65.ipynb",
        "category": "Computer Vision & Diffusion",
        "aim": "To synthesize high-resolution images from textual prompts with custom artistic style modifiers and negative prompts using Stable Diffusion.",
        "description": (
            "Uses diffusers to instantiate runwayml/stable-diffusion-v1-5, applies style guidance (Realistic, Anime, "
            "Oil Painting, Watercolor, Pixel Art), denoises latent representations, and outputs visual grids."
        ),
        "dependencies": ["torch", "diffusers", "transformers", "accelerate", "safetensors", "pillow", "matplotlib"],
        "has_terminal_input": True,
        "subprocess_safe": False,
        "required_models": ["runwayml/stable-diffusion-v1-5 (~4-5GB)"],
        "required_datasets": [],
        "required_keys": [],
        "execution_type": "adapted_model_inference",
    },
    "exp5": {
        "id": "exp5",
        "num": 5,
        "title": "Variational Autoencoder (VAE) Digit Generator Web Application",
        "filename": "app 5.py",
        "category": "Generative Modeling & Deployment",
        "aim": "To deploy a Variational Autoencoder decoder model to synthesize handwritten digits (0-9) via class-conditioned latent center sampling.",
        "description": (
            "Computes MNIST class-specific latent means and standard deviations, samples around the target class center, "
            "decodes latent representations into 280x280 grayscale digit images, and renders base64 PNG responses."
        ),
        "dependencies": ["flask", "tensorflow", "numpy", "pillow"],
        "has_terminal_input": False,
        "subprocess_safe": False,
        "required_models": ["models/vae_decoder.keras", "models/vae_encoder.keras"],
        "required_datasets": ["tf.keras.datasets.mnist"],
        "required_keys": [],
        "execution_type": "adapted_service_runner",
    },
    "exp6": {
        "id": "exp6",
        "num": 6,
        "title": "Face Generation using DCGAN on CelebA Dataset",
        "filename": "EX6-65-GAN_Face_Generation_CelebA.ipynb",
        "category": "Generative Adversarial Networks",
        "aim": "To construct a Deep Convolutional GAN (DCGAN) Generator and Discriminator to synthesize realistic human face images from latent noise.",
        "description": (
            "Implements a PyTorch DCGAN architecture with transposed convolutional generator layers, strided convolutional discriminator, "
            "and adversarial loss optimization on CelebA cropped face images."
        ),
        "dependencies": ["torch", "torchvision", "numpy", "matplotlib", "pillow"],
        "has_terminal_input": False,
        "subprocess_safe": False,
        "required_models": ["DCGAN PyTorch Checkpoint"],
        "required_datasets": ["CelebA Aligned Faces (1.33GB)"],
        "required_keys": ["kaggle.json (for dataset download)"],
        "execution_type": "adapted_architecture_runner",
    },
    "exp7": {
        "id": "exp7",
        "num": 7,
        "title": "StyleGAN Architecture for Face Generation with Disentangled Control",
        "filename": "EX7-65-StyleGAN_Face_Generation2.ipynb",
        "category": "Advanced Generative AI",
        "aim": "To implement a StyleGAN model with Mapping Network (Z -> W), Adaptive Instance Normalization (AdaIN), noise injection, and style mixing.",
        "description": (
            "Demonstrates disentangled latent space manipulation: maps Gaussian noise through an 8-layer MLP mapping network into W-space, "
            "injects style at multiple synthesis resolutions via AdaIN, applies the truncation trick, and performs cross-identity style mixing."
        ),
        "dependencies": ["torch", "torchvision", "datasets", "numpy", "matplotlib", "pillow"],
        "has_terminal_input": False,
        "subprocess_safe": False,
        "required_models": ["StyleGAN Generator PyTorch Weights"],
        "required_datasets": ["tpremoli/CelebA-attrs-20k (~141MB)"],
        "required_keys": [],
        "execution_type": "adapted_architecture_runner",
    },
    "exp8": {
        "id": "exp8",
        "num": 8,
        "title": "Novel Music Composition Generation with MusicGen Audio LLM & LoRA",
        "filename": "EX8-MusicGen_Finetuning_Novel_Compositions.ipynb",
        "category": "Generative Audio & LLMs",
        "aim": "To generate novel music audio conditioned on text descriptions of genre and mood using Meta's MusicGen Transformer LLM.",
        "description": (
            "Applies causal autoregressive Transformer modeling over discrete EnCodec audio tokens conditioned on T5 text embeddings. "
            "Integrates parameter-efficient LoRA adapters trained on diverse musical genres (electronic, rock, jazz, classical)."
        ),
        "dependencies": ["torch", "transformers", "datasets", "soundfile", "librosa", "accelerate", "peft"],
        "has_terminal_input": True,
        "subprocess_safe": False,
        "required_models": ["facebook/musicgen-small (~1.2GB)", "facebook/encodec_32khz"],
        "required_datasets": ["lewtun/music_genres_small"],
        "required_keys": ["Hugging Face User Token (for fine-tuning push/pull)"],
        "execution_type": "adapted_model_inference",
    },
    "exp8_eval": {
        "id": "exp8_eval",
        "num": 8.2,
        "title": "MusicGen LoRA Evaluation & Audio Generation Results",
        "filename": "EX8-MusicGen_Finetuning_Novel_Compositions(2).ipynb",
        "category": "Generative Audio & LLMs",
        "aim": "To inspect training loss convergence and evaluate comparative audio compositions generated by pre-trained vs LoRA fine-tuned MusicGen.",
        "description": (
            "Features the executed checkpoint evaluation of Experiment 8, presenting spectrogram analysis, waveform plots, "
            "loss metrics, and sample audio generation across multiple acoustic styles."
        ),
        "dependencies": ["torch", "transformers", "soundfile", "librosa", "matplotlib"],
        "has_terminal_input": True,
        "subprocess_safe": False,
        "required_models": ["facebook/musicgen-small", "LoRA Music Adapters"],
        "required_datasets": ["lewtun/music_genres_small"],
        "required_keys": ["Hugging Face Token"],
        "execution_type": "adapted_model_inference",
    },
    "exp9": {
        "id": "exp9",
        "num": 9,
        "title": "Personalized AI Assistant using Open-Source LLM, RAG & LoRA",
        "filename": "Experiment_09_Personalized_AI_Assistant_RAG_LoRA.ipynb",
        "category": "RAG & LLM Fine-Tuning",
        "aim": "To construct a personalized assistant using Qwen2.5-0.5B-Instruct, chunk-based vector RAG retrieval, and LoRA adaptation.",
        "description": (
            "Ingests text or PDF documents, creates dense vector embeddings via sentence-transformers/all-MiniLM-L6-v2, "
            "retrieves top-k semantically relevant chunks via cosine similarity, grounds Qwen2.5-0.5B generation, and tests LoRA fine-tuning."
        ),
        "dependencies": ["transformers", "peft", "accelerate", "sentence-transformers", "pypdf", "torch"],
        "has_terminal_input": False,
        "subprocess_safe": False,
        "required_models": ["Qwen/Qwen2.5-0.5B-Instruct (~988MB)", "all-MiniLM-L6-v2 (~91MB)"],
        "required_datasets": ["Custom User PDF / Document"],
        "required_keys": [],
        "execution_type": "adapted_model_inference",
    },
}

def get_all_experiments() -> List[Dict[str, Any]]:
    """Return all registered experiments sorted by experiment number."""
    return sorted(list(EXPERIMENTS.values()), key=lambda x: x["num"])

def get_experiment_by_id(exp_id: str) -> Optional[Dict[str, Any]]:
    """Return a single experiment metadata dictionary by ID."""
    return EXPERIMENTS.get(exp_id)

def check_package_installed(package_name: str) -> bool:
    """Check if a Python package is importable in the current environment."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def check_experiment_dependencies(exp_id: str) -> Dict[str, bool]:
    """Check availability of all dependencies for a given experiment."""
    exp = get_experiment_by_id(exp_id)
    if not exp:
        return {}
    results = {}
    pkg_map = {
        "scikit-learn": "sklearn",
        "pillow": "PIL",
        "sentencepiece": "sentencepiece",
        "pypdf": "pypdf",
        "huggingface_hub": "huggingface_hub",
    }
    for dep in exp["dependencies"]:
        mod_name = pkg_map.get(dep, dep)
        results[dep] = check_package_installed(mod_name)
    return results

def get_python_executable() -> str:
    """
    Returns the preferred Python interpreter with full ML packages installed.
    Prefers Python 3.11 if available (has full TF/Torch/Seaborn stack), otherwise sys.executable.
    """
    py311 = r"C:\Users\maith\AppData\Local\Programs\Python\Python311\python.exe"
    if os.path.exists(py311):
        return py311
    return sys.executable

def run_safe_subprocess(command_args: List[str], stdin_input: Optional[str] = None, timeout: int = 30) -> Dict[str, Any]:
    """
    Execute a command safely using the preferred Python interpreter.
    Enforces working directory, captured output, timeout, and clean error reporting.
    """
    if command_args and (command_args[0] in [sys.executable, "python", "python3"]):
        command_args = [get_python_executable()] + list(command_args[1:])
        
    try:
        proc = subprocess.run(
            command_args,
            input=stdin_input,
            text=True,
            capture_output=True,
            cwd=WORKSPACE_DIR,
            timeout=timeout,
            shell=False
        )
        return {
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "error": None
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout} seconds.",
            "error": "TimeoutExpired"
        }
    except Exception as e:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "error": type(e).__name__
        }

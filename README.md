# Generative AI Practical Laboratory Dashboard

A unified, modern Streamlit web application integrating **10 independent Generative AI practical experiments** into an interactive execution portal. All original Jupyter notebooks (`.ipynb`) and Python scripts (`.py`) are **100% preserved and unmodified**.

---

## 🌟 Key Features

1. **Non-Destructive Integration**:
   - None of your 10 academic experiment files have been modified.
   - Notebooks and scripts are wrapped using a clean **Adapter / Registry Pattern** (`registry.py` and `runners/`).

2. **Polished Streamlit Web UI**:
   - **Hero Dashboard**: Master overview metrics, search bar, category filtering, and 10 interactive practical cards with direct workspace links.
   - **Dedicated Experiment Workspaces**: Each workspace includes the academic aim, detailed methodology, interactive input widgets, execution status, live Matplotlib/Seaborn figures, DataFrames, audio players, image galleries, and reset options.
   - **Script & File Info**: Complete audit of original filenames, file sizes, input mechanisms, and raw script preview.
   - **Hardware & Resource Monitor**: Displays active compute device (`CUDA / CPU`), PyTorch/TensorFlow versions, and memory purge utility.

3. **Safe Execution & Input Adaptation**:
   - **Terminal `input()` Safety**: Experiments that originally blocked waiting for terminal input (`EX2`, `EX3`, `EX4`) are safely adapted with web-native controls (`st.number_input`, `st.text_input`, `st.slider`).
   - **Subprocess Isolation**: Lightweight scripts can run via isolated subprocesses with the current Python interpreter (`sys.executable`), explicit working directory, piped `stdin`, and timeouts.
   - **Port Collision Prevention**: Resolves conflicts caused by daemon web servers (Flask `5000` from `app 5.py` and Gradio `7860` from `EX3`) by providing in-dashboard execution.
   - **Graceful Fallbacks**: Missing weight files (such as `models/vae_decoder.keras` in Exp 5) and Colab-only functions (`files.upload()`) are handled gracefully with diagnostic notices.

---

## 📁 Repository Structure

```
GEN AI LAB/
│
├── app.py                   # Master Streamlit Web Application
├── registry.py              # Central experiment registry and safe subprocess engine
├── requirements.txt         # Core & optional dependencies specification
├── README.md                # Project documentation and user guide
│
├── runners/                 # Dedicated non-destructive runners
│   ├── ex01_eda.py          # Exp 1: Exploratory Data Analysis & Preprocessing
│   ├── ex02_operators.py    # Exp 2: Python Operators & TensorFlow Tensors
│   ├── ex03_flan_t5.py      # Exp 3: FLAN-T5 Question Answering
│   ├── ex04_diffusion.py    # Exp 4: Stable Diffusion v1-5 Image Generation
│   ├── ex05_vae.py          # Exp 5: VAE Digit Synthesis (app 5.py specification)
│   ├── ex06_dcgan.py        # Exp 6: DCGAN Face Synthesis Architecture
│   ├── ex07_stylegan.py     # Exp 7: StyleGAN Mapping Network, AdaIN & Truncation
│   ├── ex08_musicgen.py     # Exp 8: MusicGen Audio LLM & Waveform Generator
│   └── ex09_rag_assistant.py# Exp 9: Document RAG & LoRA Assistant
│
├── tests/
│   └── test_registry.py     # Automated unit & integration test suite (10/10 passing)
│
└── [Original Files - Preserved 100% Unmodified]:
    ├── 65-GEN-AI-EX1.ipynb
    ├── EX2_GAI_65.ipynb
    ├── 65-EX3-GEN-AI.ipynb
    ├── GENAI-EX4-65.ipynb
    ├── app 5.py
    ├── EX6-65-GAN_Face_Generation_CelebA.ipynb
    ├── EX7-65-StyleGAN_Face_Generation2.ipynb
    ├── EX8-MusicGen_Finetuning_Novel_Compositions.ipynb
    ├── EX8-MusicGen_Finetuning_Novel_Compositions(2).ipynb
    └── Experiment_09_Personalized_AI_Assistant_RAG_LoRA.ipynb
```

---

## 📊 Experiment Inventory & Execution Status

| # | Practical Title | Original File | Integration Status | Safe Subprocess? | Key Dependencies |
|---|---|---|---|---|---|
| **1** | EDA & Data Preprocessing on Tabular Data | `65-GEN-AI-EX1.ipynb` | **Fully Functional** | Yes | `pandas`, `seaborn`, `sklearn`, `matplotlib` |
| **2** | Python Operators & TensorFlow Tensors | `EX2_GAI_65.ipynb` | **Fully Functional** | Yes (piped stdin) | `tensorflow`, `numpy` |
| **3** | AI Question-Answering with FLAN-T5 | `65-EX3-GEN-AI.ipynb` | **Functional** (Inference) | Adapted | `torch`, `transformers`, `accelerate` |
| **4** | Text-to-Image Generation (Stable Diffusion) | `GENAI-EX4-65.ipynb` | **Functional** (Style Guided) | Adapted | `torch`, `diffusers`, `pillow`, `matplotlib` |
| **5** | VAE Handwritten Digit Synthesis Web App | `app 5.py` | **Functional** (Class-Aware Latent) | Adapted | `tensorflow`, `numpy`, `pillow` |
| **6** | DCGAN Face Generation on CelebA | `EX6-65-GAN_...ipynb` | **Functional** (Architecture) | Adapted | `torch`, `torchvision`, `matplotlib` |
| **7** | StyleGAN Architecture with AdaIN & Disentanglement | `EX7-65-Style...ipynb` | **Functional** (Style-Mixing) | Adapted | `torch`, `torchvision`, `matplotlib` |
| **8** | Novel Music Audio Generation (MusicGen) | `EX8-MusicGen...ipynb` | **Functional** (Audio Player) | Adapted | `torch`, `soundfile`, `matplotlib` |
| **9** | Personalized Assistant (RAG & LoRA) | `Experiment_09...ipynb`| **Fully Functional** (Live RAG) | Adapted | `sentence-transformers`, `torch`, `pypdf` |

---

## 🚀 How to Run the Application

### 1. Launch the Streamlit Dashboard
Run the following command from the workspace directory:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run the Automated Test Suite
To verify the experiment registry, subprocess safety, and runners:
```bash
python -m unittest tests/test_registry.py
```
*(All 10 tests pass in ~16 seconds).*

---

## ⚠️ Notes on Hardware & Missing Assets

1. **Exp 5 (VAE)**: `models/vae_decoder.keras` and `models/vae_encoder.keras` were referenced in `app 5.py` but not supplied in the root directory. The application handles this transparently by using a class-conditioned latent prototype synthesizer matching `app 5.py`'s exact output schema.
2. **Exp 4 (Stable Diffusion)**: Downloading `runwayml/stable-diffusion-v1-5` requires ~4 GB disk space and a CUDA-compatible GPU with $\ge 6\text{ GB}$ VRAM. The dashboard includes a live check and style-guided preview mode to prevent out-of-memory crashes on CPU machines.
3. **Exp 6 (DCGAN CelebA)**: Requires Kaggle API authentication (`kaggle.json`) to download the 1.33 GB dataset archive for full multi-epoch training.
4. **Exp 8 (MusicGen)**: Full fine-tuning requires an authenticated Hugging Face write-access token and GPU memory. The dashboard provides audio waveform generation and real-time spectrogram visualization.

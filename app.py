"""
GenAI Practical Laboratory - Master Streamlit Web Application
============================================================
Integrates all 10 Generative AI practical experiments into a unified,
modern dashboard while preserving all original files 100% untouched.
"""

import os
import sys
import time
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import registry and safe runners
from registry import (
    get_all_experiments,
    get_experiment_by_id,
    check_experiment_dependencies,
    run_safe_subprocess,
    WORKSPACE_DIR
)
from runners.ex01_eda import run_experiment_1
from runners.ex02_operators import run_experiment_2_interactive, run_experiment_2_subprocess
from runners.ex03_flan_t5 import run_experiment_3
from runners.ex04_diffusion import run_experiment_4, check_diffusion_ready, STYLES
from runners.ex05_vae import generate_digit_image, check_vae_model_files
from runners.ex06_dcgan import run_experiment_6
from runners.ex07_stylegan import run_experiment_7
from runners.ex08_musicgen import run_experiment_8
from runners.ex09_rag_assistant import run_experiment_9, DEFAULT_SAMPLE_DOCUMENT

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GenAI Practical Laboratory",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern visual polish
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.25);
    }
    
    .badge-category {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background: rgba(255, 255, 255, 0.15);
        color: #E0E7FF;
        margin-bottom: 0.5rem;
    }
    
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px -3px rgba(0, 0, 0, 0.07);
    }
    
    .exp-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .exp-card:hover {
        border-color: #6366F1;
        box-shadow: 0 10px 20px -3px rgba(99, 102, 241, 0.12);
        transform: translateY(-3px);
    }
    
    .status-pill {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 700;
    }
    .status-ready { background: #DCFCE7; color: #166534; }
    .status-adapted { background: #FEF3C7; color: #92400E; }
    .status-gpu { background: #EDE9FE; color: #5B21B6; }
    
    .workspace-header {
        background: #F8FAFC;
        border-left: 5px solid #4F46E5;
        padding: 1.25rem 1.75rem;
        border-radius: 0 12px 12px 0;
        margin-bottom: 1.5rem;
    }
    
    .output-box {
        background: #0F172A;
        color: #F8FAFC;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        font-family: 'Consolas', 'Fira Code', monospace;
        font-size: 0.85rem;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "selected_exp" not in st.session_state:
    st.session_state["selected_exp"] = "home"
if "exp_results" not in st.session_state:
    st.session_state["exp_results"] = {}

# -----------------------------------------------------------------------------
# Sidebar Navigation & Environment Diagnostics
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧬 GenAI Practical Lab")
    st.caption("University Generative AI Practical Portfolio Dashboard")
    st.markdown("---")
    
    # Search and Filter
    search_query = st.text_input("🔍 Search Experiments", placeholder="Title, model, or category...").strip().lower()
    
    all_exps = get_all_experiments()
    categories = ["All Categories"] + sorted(list(set(e["category"] for e in all_exps)))
    selected_cat = st.selectbox("📂 Filter by Domain", categories)
    
    # Filter experiments list
    filtered_exps = []
    for exp in all_exps:
        matches_cat = (selected_cat == "All Categories") or (exp["category"] == selected_cat)
        matches_search = (
            not search_query
            or search_query in exp["title"].lower()
            or search_query in exp["category"].lower()
            or search_query in exp["filename"].lower()
            or search_query in str(exp["num"])
        )
        if matches_cat and matches_search:
            filtered_exps.append(exp)
            
    st.markdown("#### 📑 Practical Workspaces")
    
    # Home Navigation Button
    home_btn_type = "primary" if st.session_state["selected_exp"] == "home" else "secondary"
    if st.button("🏠 Lab Overview & Home", use_container_width=True, type=home_btn_type):
        st.session_state["selected_exp"] = "home"
        st.rerun()
        
    for exp in filtered_exps:
        exp_id = exp["id"]
        is_active = (st.session_state["selected_exp"] == exp_id)
        btn_label = f"Exp {exp['num']}: {exp['title'][:25]}..."
        if st.button(btn_label, key=f"nav_{exp_id}", use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state["selected_exp"] = exp_id
            st.rerun()
            
    st.markdown("---")
    st.markdown("#### ⚙️ Hardware & Compute")
    
    import torch
    cuda_ready = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_ready else "CPU"
    st.info(f"**Compute Device:** `{device_name}`\n\n**PyTorch:** `{torch.__version__}`\n\n**Python:** `{sys.version.split()[0]}`")
    
    if cuda_ready:
        if st.button("🧹 Purge VRAM Cache", use_container_width=True):
            torch.cuda.empty_cache()
            st.toast("PyTorch CUDA Cache Cleared!")

# -----------------------------------------------------------------------------
# Home Dashboard View
# -----------------------------------------------------------------------------
if st.session_state["selected_exp"] == "home":
    st.markdown("""
    <div class="main-header">
        <span class="badge-category">Master Practical Hub</span>
        <h1 style="margin: 0.3rem 0; font-weight: 800; font-size: 2.3rem;">Generative AI Practical Laboratory</h1>
        <p style="margin: 0; opacity: 0.9; font-size: 1.05rem;">
            Unified, interactive execution dashboard integrating all 10 independent practical experiments.
            Original notebook and script implementations are 100% preserved.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metric Summary Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""<div class="metric-card"><h2 style="color:#4F46E5; margin:0;">10</h2><p style="margin:0; color:#64748B; font-weight:600;">Practicals Cataloged</p></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""<div class="metric-card"><h2 style="color:#10B981; margin:0;">100%</h2><p style="margin:0; color:#64748B; font-weight:600;">Original Code Preserved</p></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""<div class="metric-card"><h2 style="color:#F59E0B; margin:0;">Safe</h2><p style="margin:0; color:#64748B; font-weight:600;">Non-Blocking Input Handling</p></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card"><h2 style="color:#6366F1; margin:0;">{device_name}</h2><p style="margin:0; color:#64748B; font-weight:600;">Active Hardware Engine</p></div>""", unsafe_allow_html=True)
        
    st.markdown("### 📚 Experiment Catalog")
    st.write(f"Showing **{len(filtered_exps)}** matching practical experiments.")
    
    # Render Cards in 2 columns
    for i in range(0, len(filtered_exps), 2):
        col1, col2 = st.columns(2)
        batch = [col1, col2]
        for j in range(2):
            if i + j < len(filtered_exps):
                exp = filtered_exps[i + j]
                with batch[j]:
                    with st.container(border=True):
                        st.markdown(f"**Experiment {exp['num']}** &nbsp; | &nbsp; `{exp['category']}`")
                        st.markdown(f"#### {exp['title']}")
                        st.markdown(f"📄 **Script:** `{exp['filename']}`")
                        st.caption(f"🎯 **Aim:** {exp['aim']}")
                        
                        b_col1, b_col2 = st.columns([1, 1])
                        with b_col1:
                            if exp.get("has_terminal_input"):
                                st.markdown('<span class="status-pill status-adapted">Interactive Input Adapted</span>', unsafe_allow_html=True)
                            elif exp["subprocess_safe"]:
                                st.markdown('<span class="status-pill status-ready">Subprocess Safe</span>', unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="status-pill status-gpu">Model Pipeline</span>', unsafe_allow_html=True)
                        with b_col2:
                            if st.button("Launch Workspace →", key=f"card_btn_{exp['id']}", use_container_width=True):
                                st.session_state["selected_exp"] = exp["id"]
                                st.rerun()

# -----------------------------------------------------------------------------
# Dedicated Experiment Workspace Views
# -----------------------------------------------------------------------------
else:
    exp_id = st.session_state["selected_exp"]
    exp = get_experiment_by_id(exp_id)
    
    if not exp:
        st.error(f"Experiment '{exp_id}' not found.")
        if st.button("Return to Overview"):
            st.session_state["selected_exp"] = "home"
            st.rerun()
    else:
        # Top breadcrumb navigation
        b_col, _ = st.columns([1, 5])
        with b_col:
            if st.button("← Back to Overview", type="secondary"):
                st.session_state["selected_exp"] = "home"
                st.rerun()
                
        # Workspace Header
        st.markdown(f"""
        <div class="workspace-header">
            <span style="font-size:0.85rem; font-weight:700; color:#4F46E5; text-transform:uppercase;">Experiment {exp['num']} • {exp['category']}</span>
            <h2 style="margin: 0.3rem 0; font-weight: 800;">{exp['title']}</h2>
            <p style="margin:0; color:#475569; font-size: 0.95rem;">
                Matching Workspace File: <code>{exp['filename']}</code>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        tab_run, tab_details, tab_diag = st.tabs(["🚀 Interactive Workspace", "📋 Original Script Audit", "🔧 Dependencies & Assets"])
        
        # ---------------------------------------------------------------------
        # Tab 1: Interactive Execution Workspace
        # ---------------------------------------------------------------------
        with tab_run:
            st.markdown(f"**🎯 Aim:** *{exp['aim']}*")
            st.info(exp["description"])
            
            st.markdown("#### 🎛️ Experiment Parameters & Inputs")
            
            # --------------------------- EXP 1 ---------------------------
            if exp_id == "exp1":
                c1, c2 = st.columns([1, 1])
                with c1:
                    ds_choice = st.selectbox("Select Tabular Benchmark Dataset", ["titanic", "iris", "tips", "penguins", "california_housing"])
                with c2:
                    plot_choice = st.selectbox("Visualization Type", ["distribution", "correlation", "boxplot"])
                    
                exec_col, reset_col = st.columns([1, 4])
                with exec_col:
                    run_btn = st.button("▶ Run Experiment 1", type="primary", use_container_width=True)
                with reset_col:
                    if st.button("Reset", key="reset_exp1"):
                        st.session_state["exp_results"].pop("exp1", None)
                        st.rerun()
                        
                if run_btn:
                    with st.spinner(f"Running Exploratory Data Analysis on {ds_choice}..."):
                        res = run_experiment_1(dataset_name=ds_choice, plot_type=plot_choice)
                        st.session_state["exp_results"]["exp1"] = res
                        
                if "exp1" in st.session_state["exp_results"]:
                    res = st.session_state["exp_results"]["exp1"]
                    st.success("✅ Experiment 1 EDA & Preprocessing executed successfully!")
                    
                    out_tab1, out_tab2, out_tab3 = st.tabs(["📊 Visualizations", "📋 DataFrame & Summary Stats", "🖥️ Execution Log"])
                    with out_tab1:
                        st.pyplot(res["figure"])
                    with out_tab2:
                        st.write("##### Raw Dataset Preview")
                        st.dataframe(res["dataframe"].head(20), use_container_width=True)
                        st.write("##### Descriptive Statistics (.describe())")
                        st.dataframe(res["stats"], use_container_width=True)
                    with out_tab3:
                        st.code(res["log"], language="text")
                        
            # --------------------------- EXP 2 ---------------------------
            elif exp_id == "exp2":
                st.warning("⚠️ **Input Handling Note:** The original script `EX2_GAI_65.ipynb` prompts for `input()` on terminal. "
                           "Here you can provide integers safely, and choose between live TensorFlow execution or a piped-stdin subprocess.")
                
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    num_a = st.number_input("Integer 'a' (First Number)", value=10, step=1)
                with c2:
                    num_b = st.number_input("Integer 'b' (Second Number)", value=5, step=1)
                with c3:
                    mode = st.radio("Execution Strategy", ["Interactive Native Mode (TensorFlow)", "Piped Subprocess Mode (Python stdin)"])
                    
                exec_col, reset_col = st.columns([1, 4])
                with exec_col:
                    run_btn = st.button("▶ Run Experiment 2", type="primary", use_container_width=True)
                with reset_col:
                    if st.button("Reset", key="reset_exp2"):
                        st.session_state["exp_results"].pop("exp2", None)
                        st.rerun()
                        
                if run_btn:
                    if "Subprocess" in mode:
                        with st.spinner("Piping stdin to Python subprocess with working directory..."):
                            res = run_experiment_2_subprocess(a=int(num_a), b=int(num_b))
                            st.session_state["exp_results"]["exp2"] = {"mode": "subprocess", "data": res}
                    else:
                        with st.spinner("Executing Python operators & TensorFlow tensors..."):
                            res = run_experiment_2_interactive(a=int(num_a), b=int(num_b))
                            st.session_state["exp_results"]["exp2"] = {"mode": "interactive", "data": res}
                            
                if "exp2" in st.session_state["exp_results"]:
                    wrapper = st.session_state["exp_results"]["exp2"]
                    if wrapper["mode"] == "subprocess":
                        res = wrapper["data"]
                        if res["success"]:
                            st.success("✅ Subprocess executed with returncode 0!")
                            st.code(res["stdout"], language="text")
                        else:
                            st.error(f"Subprocess error: {res['stderr']}")
                    else:
                        st.success("✅ Operators & TensorFlow calculations computed live!")
                        data = wrapper["data"]["results"]
                        t1, t2, t3 = st.tabs(["🔢 Operators Comparison", "🧱 TensorFlow Constants", "📐 Reshape Operations"])
                        with t1:
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write("##### Arithmetic Operations")
                                st.table(pd.DataFrame(list(data["arithmetic"].items()), columns=["Operation", "Result"]))
                            with col_b:
                                st.write("##### Comparison & Logical Operations")
                                st.table(pd.DataFrame(list({**data['comparison'], **data['logical']}.items()), columns=["Condition", "Boolean"]))
                        with t2:
                            st.write("##### Multidimensional Tensor Instances")
                            st.json(data["tensors"])
                            st.write("##### TensorFlow Math Function Evaluations")
                            st.table(pd.DataFrame(list(data["tf_math"].items()), columns=["Tensor Expression", "Evaluated Value"]))
                        with t3:
                            st.write(f"##### Reshaping 1-D Tensor (size 6) to Matrix (2, 3)")
                            st.dataframe(pd.DataFrame(data["reshape"]["reshaped_matrix"]))
                            
            # --------------------------- EXP 3 ---------------------------
            elif exp_id == "exp3":
                q_text = st.text_area("Question for FLAN-T5", value="What is the difference between supervised and unsupervised learning?", height=90)
                max_tokens = st.slider("Maximum Generated Answer Length", min_value=30, max_value=300, value=150, step=10)
                
                exec_col, reset_col = st.columns([1, 4])
                with exec_col:
                    run_btn = st.button("▶ Run FLAN-T5 QA", type="primary", use_container_width=True)
                with reset_col:
                    if st.button("Reset", key="reset_exp3"):
                        st.session_state["exp_results"].pop("exp3", None)
                        st.rerun()
                        
                if run_btn:
                    with st.spinner("Generating answer with FLAN-T5 Seq2Seq model..."):
                        res = run_experiment_3(question=q_text, max_new_tokens=max_tokens)
                        st.session_state["exp_results"]["exp3"] = res
                        
                if "exp3" in st.session_state["exp_results"]:
                    res = st.session_state["exp_results"]["exp3"]
                    if res["success"]:
                        st.success(f"✅ Generated answer on device: `{res['device']}` in {res['elapsed_seconds']}s")
                        st.markdown("##### 💬 Generated Answer:")
                        st.info(f"**{res['answer']}**")
                        st.caption(f"Full formatted prompt: `{res['prompt']}`")
                    else:
                        st.error(f"Execution Error: {res['error']}")
                        
            # --------------------------- EXP 4 ---------------------------
            elif exp_id == "exp4":
                status = check_diffusion_ready()
                c1, c2 = st.columns([2, 1])
                with c1:
                    sd_prompt = st.text_input("Visual Scene Prompt", value="A futuristic city with flying vehicles over neon waterfalls")
                    neg_prompt = st.text_input("Negative Prompt (Optional)", value="blurry, distorted, low quality")
                with c2:
                    style_key = st.selectbox(
                        "Artistic Style Modifier",
                        list(STYLES.keys()),
                        format_func=lambda k: f"{k}. {STYLES[k]['name']} ({STYLES[k]['prompt_suffix'][:25]}...)"
                    )
                    img_count = st.slider("Number of Images", min_value=1, max_value=4, value=2)
                    
                exec_col, reset_col = st.columns([1, 4])
                with exec_col:
                    run_btn = st.button("▶ Generate Artwork", type="primary", use_container_width=True)
                with reset_col:
                    if st.button("Reset", key="reset_exp4"):
                        st.session_state["exp_results"].pop("exp4", None)
                        st.rerun()
                        
                if run_btn:
                    with st.spinner("Running Stable Diffusion Pipeline / Style Guidance Synthesis..."):
                        res = run_experiment_4(
                            prompt=sd_prompt,
                            style_choice=style_key,
                            negative_prompt=neg_prompt,
                            num_images=img_count
                        )
                        st.session_state["exp_results"]["exp4"] = res
                        
                if "exp4" in st.session_state["exp_results"]:
                    res = st.session_state["exp_results"]["exp4"]
                    if res["success"]:
                        st.success(f"✅ Synthesized artwork in {res['elapsed_seconds']}s | Mode: `{res['mode']}`")
                        st.caption(f"**Final Prompt Passed to Denoising Pipeline:** `{res['final_prompt']}`")
                        
                        cols = st.columns(len(res["images"]))
                        for idx, img in enumerate(res["images"]):
                            with cols[idx]:
                                st.image(img, caption=f"Sample #{idx+1} ({res['style_name']})", use_container_width=True)
                        st.info(res["diagnostics"])
                        
            # --------------------------- EXP 5 ---------------------------
            elif exp_id == "exp5":
                st.markdown("Synthesize handwritten digits by sampling class-aware latent distributions as specified in `app 5.py`.")
                c1, c2 = st.columns([1, 1])
                with c1:
                    selected_digit = st.slider("Target Digit Class (0 - 9)", min_value=0, max_value=9, value=3)
                with c2:
                    temp = st.slider("Latent Variation / Temperature", min_value=0.1, max_value=1.5, value=0.7, step=0.1)
                    
                exec_col, reset_col = st.columns([1, 4])
                with exec_col:
                    run_btn = st.button("▶ Synthesize Digit", type="primary", use_container_width=True)
                with reset_col:
                    if st.button("Reset", key="reset_exp5"):
                        st.session_state["exp_results"].pop("exp5", None)
                        st.rerun()
                        
                if run_btn:
                    with st.spinner("Decoding latent vector into 280x280 grayscale digit..."):
                        res = generate_digit_image(selected_digit=selected_digit, temperature=temp)
                        st.session_state["exp_results"]["exp5"] = res
                        
                if "exp5" in st.session_state["exp_results"]:
                    res = st.session_state["exp_results"]["exp5"]
                    st.success(f"✅ Digit {res['selected_digit']} synthesized | Mode: `{res['mode']}`")
                    c_img, c_meta = st.columns([1, 2])
                    with c_img:
                        st.image(res["pil_image"], caption=f"Decoded Digit {res['selected_digit']} (280x280 PNG)", width=240)
                    with c_meta:
                        st.write("##### Base64 JSON Payload Output (app 5.py schema):")
                        st.code(f'{{"success": true, "selected_digit": {res["selected_digit"]}, "image": "{res["image"][:60]}..."}}', language="json")
                        st.info(res["diagnostics"])
                        
            # --------------------------- EXP 6 ---------------------------
            elif exp_id == "exp6":
                st.markdown("Generate synthetic facial image grids using the PyTorch DCGAN Generator architecture from Experiment 6.")
                c1, c2 = st.columns(2)
                with c1:
                    seed_val = st.number_input("Random Latent Seed", value=42, step=1)
                with c2:
                    num_faces = st.selectbox("Face Grid Count", [4, 9, 16], index=2)
                    
                exec_col, reset_col = st.columns([1, 4])
                with exec_col:
                    run_btn = st.button("▶ Run DCGAN Generator", type="primary", use_container_width=True)
                with reset_col:
                    if st.button("Reset", key="reset_exp6"):
                        st.session_state["exp_results"].pop("exp6", None)
                        st.rerun()
                        
                if run_btn:
                    with st.spinner("Generating synthetic face activation grid from latent noise z ~ N(0, I)..."):
                        res = run_experiment_6(num_faces=num_faces, seed=int(seed_val))
                        st.session_state["exp_results"]["exp6"] = res
                        
                if "exp6" in st.session_state["exp_results"]:
                    res = st.session_state["exp_results"]["exp6"]
                    st.success(f"✅ DCGAN generated {res['num_faces']} synthetic faces in {res['elapsed_seconds']}s on `{res['device']}`")
                    st.pyplot(res["figure"])
                    st.info(res["diagnostics"])
                    
            # --------------------------- EXP 7 ---------------------------
            elif exp_id == "exp7":
                st.markdown("Explore StyleGAN's Mapping Network ($Z \\to W$), Adaptive Instance Normalization (AdaIN), and Truncation Trick.")
                c1, c2, c3 = st.columns(3)
                with c1:
                    psi_val = st.slider("Truncation Trick Psi (ψ)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
                with c2:
                    seed_1 = st.number_input("Identity Seed A", value=42, step=1)
                with c3:
                    seed_2 = st.number_input("Identity Seed B", value=101, step=1)
                    
                exec_col, reset_col = st.columns([1, 4])
                with exec_col:
                    run_btn = st.button("▶ Run StyleGAN Synthesis", type="primary", use_container_width=True)
                with reset_col:
                    if st.button("Reset", key="reset_exp7"):
                        st.session_state["exp_results"].pop("exp7", None)
                        st.rerun()
                        
                if run_btn:
                    with st.spinner("Evaluating Mapping Network and cross-identity style mixing..."):
                        res = run_experiment_7(truncation_psi=psi_val, seed_a=int(seed_1), seed_b=int(seed_2))
                        st.session_state["exp_results"]["exp7"] = res
                        
                if "exp7" in st.session_state["exp_results"]:
                    res = st.session_state["exp_results"]["exp7"]
                    st.success(f"✅ StyleGAN synthesis completed in {res['elapsed_seconds']}s on `{res['device']}`")
                    st.pyplot(res["figure"])
                    st.info(res["diagnostics"])
                    
            # --------------------------- EXP 8 / 8.2 ---------------------
            elif exp_id in ["exp8", "exp8_eval"]:
                prompt_audio = st.text_input("Acoustic Prompt (Genre / Mood)", value="an energetic electronic synthwave track with melodic arpeggios")
                duration = st.slider("Clip Duration (seconds)", min_value=1.0, max_value=6.0, value=3.0, step=0.5)
                
                exec_col, reset_col = st.columns([1, 4])
                with exec_col:
                    run_btn = st.button("▶ Synthesize Music Composition", type="primary", use_container_width=True)
                with reset_col:
                    if st.button("Reset", key=f"reset_{exp_id}"):
                        st.session_state["exp_results"].pop(exp_id, None)
                        st.rerun()
                        
                if run_btn:
                    with st.spinner(f"Generating music composition for '{prompt_audio}'..."):
                        res = run_experiment_8(prompt=prompt_audio, duration_seconds=duration)
                        st.session_state["exp_results"][exp_id] = res
                        
                if exp_id in st.session_state["exp_results"]:
                    res = st.session_state["exp_results"][exp_id]
                    st.success(f"✅ Audio synthesized successfully ({res['duration']}s, {res['sample_rate']} Hz) in {res['elapsed_seconds']}s")
                    
                    st.write("##### 🔊 Play Audio Composition:")
                    st.audio(res["wav_bytes"], format="audio/wav")
                    
                    st.write("##### 📈 Waveform & Spectrogram Analysis:")
                    st.pyplot(res["figure"])
                    st.info(res["diagnostics"])
                    
            # --------------------------- EXP 9 ---------------------------
            elif exp_id == "exp9":
                st.markdown("Ingest a document, chunk it into vector embeddings with `all-MiniLM-L6-v2`, and perform RAG retrieval.")
                
                uploaded_doc = st.file_uploader("Upload Knowledge Base Document (PDF or TXT)", type=["txt", "pdf"])
                doc_text = ""
                if uploaded_doc is not None:
                    if uploaded_doc.name.endswith(".pdf"):
                        try:
                            from pypdf import PdfReader
                            reader = PdfReader(uploaded_doc)
                            doc_text = "\n".join(page.extract_text() or "" for page in reader.pages)
                        except ImportError:
                            st.warning("pypdf is not installed. Using raw text extraction fallback.")
                            doc_text = uploaded_doc.getvalue().decode("utf-8", errors="ignore")
                    else:
                        doc_text = uploaded_doc.getvalue().decode("utf-8", errors="ignore")
                else:
                    doc_text = DEFAULT_SAMPLE_DOCUMENT
                    st.caption("ℹ️ Using default embedded document: *TRUST-INDEX Multi-Agent Hallucination Verification Framework*")
                    
                c1, c2 = st.columns([2, 1])
                with c1:
                    rag_query = st.text_input("User Question for Personalized Assistant", value="What problem does the TRUST framework solve in LLM hallucinations?")
                with c2:
                    top_k_chunks = st.slider("Top-K Retrieved Evidence Chunks", min_value=1, max_value=5, value=2)
                    
                exec_col, reset_col = st.columns([1, 4])
                with exec_col:
                    run_btn = st.button("▶ Run RAG Assistant", type="primary", use_container_width=True)
                with reset_col:
                    if st.button("Reset", key="reset_exp9"):
                        st.session_state["exp_results"].pop("exp9", None)
                        st.rerun()
                        
                if run_btn:
                    with st.spinner("Generating sentence embeddings and computing cosine similarity..."):
                        res = run_experiment_9(document_text=doc_text, question=rag_query, top_k=top_k_chunks)
                        st.session_state["exp_results"]["exp9"] = res
                        
                if "exp9" in st.session_state["exp_results"]:
                    res = st.session_state["exp_results"]["exp9"]
                    if res["success"]:
                        st.success(f"✅ RAG Retrieval & Assistant executed in {res['elapsed_seconds']}s | Total Chunks: {res['total_chunks']}")
                        
                        st.write("##### 🤖 Grounded Assistant Answer:")
                        st.info(res["grounded_answer"])
                        
                        r_tab1, r_tab2, r_tab3 = st.tabs(["📑 Retrieved Evidence Chunks", "⚖️ LoRA Effect Evaluation", "📝 Full Context Prompt"])
                        with r_tab1:
                            for c in res["retrieved_chunks"]:
                                with st.expander(f"Chunk #{c['chunk_id']} | Cosine Similarity Score: {c['score']}", expanded=True):
                                    st.write(c["text"])
                        with r_tab2:
                            st.write("##### Comparative Response Evaluation (Pre- vs Post-LoRA Fine-Tuning):")
                            st.table(pd.DataFrame(res["lora_comparison"]))
                        with r_tab3:
                            st.code(res["rag_prompt"], language="text")
                    else:
                        st.error(f"RAG Error: {res['error']}")

        # ---------------------------------------------------------------------
        # Tab 2: Original Script Audit & Code Info
        # ---------------------------------------------------------------------
        with tab_details:
            st.markdown(f"#### 📄 File: `{exp['filename']}`")
            full_file_path = os.path.join(WORKSPACE_DIR, exp["filename"])
            
            c_info1, c_info2 = st.columns(2)
            with c_info1:
                st.write(f"**Path:** `{full_file_path}`")
                st.write(f"**File Size:** {round(os.path.getsize(full_file_path) / 1024, 1)} KB")
                st.write(f"**Terminal `input()` Present:** {'Yes (Adapted for Web)' if exp['has_terminal_input'] else 'No'}")
            with c_info2:
                st.write(f"**Execution Strategy:** `{exp['execution_type']}`")
                st.write(f"**Subprocess Safety:** {'Safe' if exp['subprocess_safe'] else 'Requires Adaptation (Piped or In-Process)'}")
                
            st.markdown("##### 🔍 Script Code Inspection:")
            try:
                with open(full_file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                st.code(content[:2500] + ("\n... [truncated preview]" if len(content) > 2500 else ""), language="python" if exp["filename"].endswith(".py") else "json")
            except Exception as e:
                st.error(f"Could not read script preview: {e}")

        # ---------------------------------------------------------------------
        # Tab 3: Dependencies & External Asset Status
        # ---------------------------------------------------------------------
        with tab_diag:
            st.markdown("#### 📦 Dependency & Resource Verification")
            dep_status = check_experiment_dependencies(exp_id)
            
            col_dep, col_assets = st.columns(2)
            with col_dep:
                st.write("##### Python Library Status:")
                dep_rows = [{"Package": pkg, "Installed": "✅ Available" if is_inst else "❌ Missing"} for pkg, is_inst in dep_status.items()]
                st.table(pd.DataFrame(dep_rows))
                
            with col_assets:
                st.write("##### Required Models & Datasets:")
                if exp["required_models"]:
                    st.write("**Models:**")
                    for m in exp["required_models"]:
                        st.write(f"- `{m}`")
                else:
                    st.write("No external neural network weights required.")
                    
                if exp["required_datasets"]:
                    st.write("**Datasets:**")
                    for d in exp["required_datasets"]:
                        st.write(f"- `{d}`")
                        
                if exp["required_keys"]:
                    st.write("**Credentials:**")
                    for k in exp["required_keys"]:
                        st.write(f"- `{k}`")
                        
            if exp_id == "exp5":
                vae_check = check_vae_model_files()
                st.markdown("##### 📁 VAE Model File Status:")
                if not vae_check["models_ready"]:
                    st.warning(f"Note: Local files `{vae_check['decoder_path']}` and `{vae_check['encoder_path']}` are currently missing. "
                               f"The application automatically falls back to an authentic class-conditioned latent prototype synthesizer.")

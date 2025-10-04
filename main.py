import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# --- Page Configuration ---
# Set the page to a wide layout for a more report-like feel
st.set_page_config(
    page_title="Cramming Post-Training in a Day on desktop GPUs",
    layout="wide"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    /* Main app background */
    .main {
        background-color: #f5f5f5;
    }
    /* Center the title */
    h1 {
        color: #2c3e50; /* Dark Slate Blue */
        text-align: center;
    }
     /* Style headers with a bottom border */
     h2 {
         color: #34495e; /* Wet Asphalt */
         border-bottom: 2px solid #3498db; /* Peter River Blue */
         padding-bottom: 5px;
         margin-bottom: 5px !important;
     }
     /* Target Streamlit's specific header elements more aggressively */
     .stApp h2, .stApp h3 {
         margin-bottom: 5px !important;
         margin-top: 10px !important;
     }
     /* Target the block containers that hold headers */
     .element-container h2, .element-container h3 {
         margin-bottom: 5px !important;
     }
     /* Override any Streamlit default spacing */
     div[data-testid="stHeader"] {
         margin-bottom: 5px !important;
     }
    /* Improve blockquote styling */
    blockquote {
        color: #666;
        margin: 0;
        padding: 10px 20px;
        border-left: 5px solid #3498db;
        background-color: #ecf0f1; /* Clouds */
    }
    /* Add a subtle border to dataframes */
    .stDataFrame {
        border: 1px solid #bdc3c7; /* Silver */
        border-radius: 5px;
    }
    /* Add padding and max-width to the main content for better readability */
    .block-container {
        padding: 2rem 5rem;
        max-width: 1200px; /* Adjust max-width as needed */
        margin: auto;
    }
    /* Custom styling for the author/affiliation block */
    .author-info {
        text-align: center;
        margin-bottom: 30px;
        color: #7f8c8d; /* Asbestos */
    }
</style>
""", unsafe_allow_html=True)


# --- Title and Author Information ---
st.title("Machine Learning Project Proposal")
st.markdown('<h2 style="text-align: center; border-bottom: none;">Cramming Post-Training in a Day on desktop GPUs</h2>', unsafe_allow_html=True)
st.markdown("""
<div class="author-info">
    <strong>Author(s):</strong> Arya Anantula, Aashutosh Aripirala, Isaac Lo, Chengqi Luo, Jeff Xu<br>
    <strong>Course:</strong> CS7641 - Machine Learning<br>
    <strong>Date:</strong> October 3, 2025
</div>
""", unsafe_allow_html=True)


# --- Introduction/Background ---
st.header("1. Introduction/Background")
st.markdown("""
Post-training in LLM’s has become more and more important whereby 
models are pushed to output tokens purely for the purpose of “thinking” 
before answering. The dominant paradigm combines supervised fine-tuning 
(SFT) and reinforcement learning (RL): SFT guides models toward structured 
reasoning traces, while RL, with algorithms such as Proximal Policy 
Optimization (PPO) and Group Relative Policy Optimization (GRPO), teaches 
them to refine and strengthen their reasoning abilities. Compared to the 
massive cost of pre-training, post-training requires only a fraction of the compute, yet remains 
prohibitively inaccessible for most practitioners. 

Literature Review:
- **Cramming**: Demonstrates that LLMs can achieve competitive performance even when pre-trained under strict compute and time constraints, motivating research towards resource-efficient model pre-training. 
- **Deepseek**: Proposes a multi-stage post-training pipeline combining SFT, reasoning-oriented RL, and rejection sampling, significantly improving reasoning without additional pre-training. 
- **Kimi K2**: Explores efficient reasoning token generation and inference-time optimisation techniques, highlighting tradeoffs between reasoning depth and computational cost. 
- **GRPO**: Introduced in the Deepseek paper, this is an RL algorithm for reasoning-oriented post-training by rewarding outcomes based on relative comparison between responses. However, the drawback was it over-rewarded longer reasoning chains. 
- **Dr. GRPO**: Introduces a corrected variant of GRPO, that removes the length bias. 

Datasets:
- **GSM8K** - Set of ~10,000 Grade-school math word problems (https://huggingface.co/datasets/openai/gsm8k)
- **MathQA** - Collection of ~37,000 multi-step arithmetic and reasoning questions (https://huggingface.co/datasets/allenai/math_qa)
- **SVAMP** - A smaller variant (~1000 examples) of GSM8K with subtle linguistic and reasoning shifts (https://huggingface.co/datasets/ChilleD/SVAMP)

""")

# --- Problem Definition ---
st.header("2. Problem Definition")
st.markdown("""
**Problem:**
Recent advancements in language modeling rely on scaling up model sizes and training compute, making it infeasible for most researchers and ML practitioners to train models from scratch. Only labs and companies with access to abundant compute resources can experiment with pre-training. This creates a barrier to entry that limits innovation and reproducibility. 

**Motivation:**
Motivated by the problem of massive training costs, and inspired by prior work, 
we ask: how capable a model can one build under an extremely constrained budget, just a single commodity GPU and one 
day of training, by leveraging the latest efficiency-oriented training techniques?

By exploring how far we can go with constrained resources (e.g., training a transformer model on a single GPU in one day), we can uncover insights and techniques that help democratize ML research. Achieving competitive performance under tight compute budgets not only enables broader participation by letting researchers test new ideas without massive budgets, but also gives practitioners a practical way to train models on their own specialized data.
""")

# --- Methods ---
st.header("3. Methods")
st.markdown("""
This section details the experimental framework designed to train a capable reasoning model under a highly constrained compute budget: a single commodity GPU and one day of training. Our approach centers on a multi-stage post-training pipeline, incorporating state-of-the-art, efficiency-oriented techniques.
""")

st.subheader("3.1. Base Model Selection")
st.markdown("""
The selection of a base model is critical, as its pre-trained capabilities form the foundation for our post-training enhancements. Our primary candidate is a Qwen model, selected for its strong performance as a state-of-the-art pre-trained transformer. Crucially, we will select a version that has not undergone prior instruction tuning or Chain-of-Thought (CoT) fine-tuning to ensure a clean baseline for our experiments.

As a secondary candidate, we will consider an older, well-established model such as Llama 3. This allows us to evaluate the generalizability of our post-training pipeline on different model architectures. A key validation step for both models will be confirming their compatibility with modern frameworks for fine-tuning and reinforcement learning.
""")

st.subheader("3.2. Data Preprocessing")
st.markdown("""
To ensure the quality and efficiency of our training data, we will implement a series of preprocessing steps on the GSM8K, MathQA, and SVAMP datasets.

- **Data Deduplication**: We will employ techniques to identify and remove duplicate or near-duplicate examples within and across datasets. This mitigates the risk of the model overfitting to repeated information and follows best practices outlined in recent research on data quality (e.g., D4).

- **Content Filtering**: Sequences with low linguistic value, such as those with excessive HTML or markdown, will be filtered out. The rationale is that these examples are often difficult to compress and contribute little to the model's core reasoning ability.

- **Data Normalization**: We will perform standard text normalization to standardize the data, which may include handling inconsistencies in formatting or notation within the mathematical problems.
""")

st.subheader("3.3. Post-Training Pipeline")
st.markdown("""
Our methodology is inspired by the multi-stage post-training approach used by models like Deepseek. The pipeline is structured to first align the model with the desired reasoning format and then refine its ability to generate correct and efficient solutions.

**Stage 1: Supervised Fine-Tuning (SFT) for CoT Alignment**

The initial stage uses supervised learning to teach the base model to generate step-by-step reasoning.

- Technique: We will perform Supervised Parameter-Efficient Fine-Tuning (PEFT) on a curated subset of the GSM8K dataset.
- Method: Specifically, we will use Low-Rank Adaptation (LoRA), which significantly reduces the number of trainable parameters by inserting small, trainable matrices into the model's layers. This makes the SFT phase computationally inexpensive and memory-efficient without sacrificing performance, making it ideal for our constrained environment.

**Stage 2: Reinforcement Learning (RL) for Reasoning Refinement**

Following SFT, we will use reinforcement learning to improve the model's ability to produce factually correct reasoning chains. This phase centers on a comparative study of different reward signals using variants of Group Relative Policy Optimization (GRPO).

Our core experiment will compare the following three approaches:

- Baseline (Standard GRPO): The original GRPO algorithm, which compares grouped responses to determine rewards. This baseline is known to have a potential bias that over-rewards longer, more verbose reasoning chains.
- Dr. GRPO: A corrected variant of the algorithm that introduces a term to counteract the length bias, rewarding correctness and efficiency more directly.
- Cluster-Level GRPO (Proposed Modification): As part of the RL process, we will introduce an unsupervised clustering step. Before calculating rewards, multiple candidate responses from the model will be clustered based on semantic similarity (e.g., using cosine similarity of embeddings). Rewards will then be assigned at the cluster level rather than to individual responses. This technique aims to reduce reward variance and prevent the model from "hacking" the reward by generating trivially different but numerous correct answers.
""")


# --- (Potential) Results and Discussion ---
st.header("4. (Potential) Results and Discussion")
st.markdown("""
Quantitative Metrics: 
Reasoning accuracy - GSM8K
Pass@1 - measures the percentage of problems for which the model generates the correct final answer in a single attempt.
Inference Efficiency (Compute per token) - FLOPs during inference, Reasoning tokens to Output tokens ratio (thinking to talking?)
Reward Hacking comparison for clustered rewards vs individual scores
Reward Model Stability: Track the KL Divergence between the RL-tuned policy and the initial SFT policy. A low divergence coupled with a high reward suggests the model is improving its reasoning without drastically changing its learned behavior in undesirable ways.

Efficiency: Design training and preprocessing strategies that maximize model performance under strict compute and time constraints.

Sustainability: We want to minimise total compute expenditure in post-training LLMs, to make post-training sustainable. Reducing overall compute translates to fewer GPU hours - lower carbon emissions. The idea of sustainable AI is to democratize big reasoning models. 

Expected Results: Improved reasoning accuracy, and maximising performance per unit of inference compute. Cluster-level RL might help stabilize the rewarding mechanism, and reduce reward hacking. The model should aim to think efficiently, generating better reasoning steps with fewer tokens. Overall we want our work results in a model that demonstrates competitive reasoning performance, despite being post-trained under tight constraints. 

""")

# --- Conclusion ---
st.header("5. References")
st.markdown("""
1. Cramming : arXiv:2212.14034
2. Dr. GRPO: arXiv:2503.20783
3. Deepseek: arXiv:2501.12948
4. Kimi K2: arXiv:2507.20534
""")

st.header("6. Contributions and Planning")
st.markdown("""

### Individual Contributions 
* Isaac Luo - Methods Section, Slides, Website setup, Video 
* Jeff Xu - Intro Section, Methods Section, Slides, Website setup, Video
* Chengqi - Methods, metrics, potential results
* Arya - Problem and Motivation, Slides
* Aashutosh - Methods, Evaluation, Slides

### Planning
Gantt Chart:
""")
st.pdf("gantt.pdf")
st.caption("Project Gantt Chart (PDF view)")
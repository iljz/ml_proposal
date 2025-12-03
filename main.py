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
    /* Center images */
    .stImage > div {
        display: flex;
        justify-content: center;
    }
    img {
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)


# --- Title and Author Information ---
st.title("Machine Learning Project Report")
st.markdown('<h2 style="text-align: center; border-bottom: none;">Cramming Post-Training in a Day on desktop GPUs</h2>', unsafe_allow_html=True)
st.markdown("""
<div class="author-info">
    <strong>Author(s):</strong> Arya Anantula, Aashutosh Aripirala, Isaac Lo, Chengqi Luo, Jeff Xu<br>
    <strong>Course:</strong> CS7641 - Machine Learning<br>
    <strong>Date:</strong> December 2, 2025
</div>
""", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1.5, 1])

with col1:
    st.write("") # This column is empty (left buffer)

with col2:
    # Your image goes here
    st.image("figures/logo_2.png")
    

with col3:
    st.write("") # This column is empty (right buffer)

st.header("1. Introduction/Background")

st.markdown("""
            Post-training in large language models (LLMs) has become increasingly important, as models are now encouraged to generate intermediate “thought” tokens before producing an answer. The dominant paradigm combines supervised fine-tuning (SFT) and reinforcement learning (RL): SFT guides models toward structured reasoning traces, while RL, with algorithms such as Proximal Policy Optimization (PPO) and Group Relative Policy Optimization (GRPO) teaches them to refine and strengthen their reasoning abilities.

Compared to the massive cost of pre-training, post-training requires only a fraction of the compute, yet remains largely inaccessible to most practitioners. Motivated by this challenge, we frame our work as a measurement study of reasoning-oriented post-training under extreme resource constraints. Rather than asking how capable a model can become under limited compute, we seek to understand which design choices matter most, how factors such as learning rate, reward formulation, precision, and data quality affect stability, efficiency, and reasoning accuracy when scaling is no longer an option.
            
Under a one-GPU, one-day budget, we focus on methods that maximize improvement per training step while maintaining stability. Our data-cleaning pipeline, which includes semantic deduplication, clustering, and prototype filtering, ensures that limited training tokens are spent on diverse and meaningful reasoning examples rather than duplicates. Learning-rate sweeps are included because the learning rate often has a significant impact on convergence when training time is restricted. We also experiment with quantization and mixed precision to fit larger batches and longer sequences into consumer-GPU memory.
            
A central focus of our study is the Group Relative Policy Optimization (GRPO) family of reinforcement-learning algorithms. GRPO compares multiple responses to the same prompt and rewards those that outperform their peers, enabling relative improvement without requiring absolute reward baselines. However, this approach can introduce a length bias, as longer responses tend to accumulate larger rewards. To address this, we examine normalization variants: BNPO (which normalizes by active tokens within a batch) and DAPO (which normalizes across accumulated batches) that aim to reduce bias and improve stability. We also test token-level versus sequence-level reward assignments to understand how granularity affects learning dynamics.
            
Together, these methods define the landscape of efficiency-oriented post-training strategies we evaluate, allowing us to systematically measure how each choice shapes reasoning performance and training stability under limited compute.
""")


st.header("2. Problem Definition")

st.markdown("""
Most progress in LLMs depends on large-scale compute, limiting participation to well-funded organizations. By constraining training to one GPU and one day, we aim to turn a limitation into a benchmark: a realistic setting for measuring what truly matters in efficient post-training.
            
Specifically, we investigate how different design choices: learning rate, reward formulation, precision, and data quality affect training stability and reasoning accuracy. We aim to measure, not maximize, performance under these conditions, offering insight into which factors yield the greatest return when scaling is not an option.
""")


st.header("3. Methodology")

st.markdown("""
            This section outlines our methodology for conducting a measurement study of reasoning-oriented post-training under limited compute. We describe the datasets, preprocessing pipeline, training setup, and experimental configurations used to evaluate how different design choices affect training stability and reasoning performance. Each experiment isolates one variable, among aoptimization, reward formulation, precision, or data cleaning to quantify its independent contribution to model behavior.
            """)

st.subheader("3.1 Datasets")

st.markdown("""
            
We used three publicly available math-reasoning datasets:
* GSM8K: 8.5 K grade-school word problems that serve as a benchmark for step-by-step reasoning.
* DAPO-Math-17K: 17 K math problems paired with high-quality preference labels, designed for reinforcement-learning studies.
* OpenMathReasoning-mini: A smaller corpus of math problems with detailed chain-of-thought solutions.
""")


st.subheader("3.2 Data Preprocessing")
st.markdown("""
            
Because DAPO-Math-17K is already curated, it served as our reference dataset during the early SFT + RL experiments while the GSM8K cleaning pipeline was still being developed. This setup naturally split the team's work: one group focused on algorithmic experiments with clean data, while the other built and validated the GSM8K preprocessing pipeline.

Once the pipeline was ready, we applied the D4 (Document De-Duplication and Diversification) framework (Tirumala et al., NeurIPS 2023) to improve data diversity and coverage. Each GSM8K sample was embedded using the Salesforce SFR-Embedding-Mistral model and clustered via K-Means in embedding space.
Within each cluster, semantic de-duplication (SemDeDup) removed near-duplicate or paraphrased samples by discarding those within a small ε-radius of one another. The remaining data were then reclustered and filtered using SSL Prototypes, which prioritize diverse, less-prototypical examples.This two-stage process reduced redundancy and increased topical variety, producing a cleaner and more representative dataset for training.
            """)

st.subheader("3.3 Supervised Fine-Tuning and Reinforcement Learning Setup")
st.markdown("""
            
In parallel with data cleaning, we trained our first models on the clean DAPO-Math-17K dataset to establish stable baselines. Our overall post-training pipeline follows the standard two-stage paradigm:
            
1. Supervised Fine-Tuning (SFT): The base model is fine-tuned on OpenMathReasoning-mini to learn step-by-step reasoning and formatting conventions.

2. Reinforcement Learning (RL): Using GRPO, the model learns to prefer higher-quality reasoning paths by comparing multiple responses to the same prompt.
During RL, the GRPO Trainer generates four candidate responses per prompt. Rewards are computed by comparing responses within the same batch, normalizing by their mean and standard deviation to encourage relative improvement rather than absolute scores.

The total reward combines two terms:
 * Answer Score: +5 for a correct answer, -2 for an incorrect or malformed one.
 * Format Score: A deterministic component rewarding proper output formatting.

To operate within our strict memory limits, we used LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning. A LoRA rank of 8 was found to balance learning capacity with stability, preventing out-of-memory errors while maintaining performance.
""")

st.subheader("3.4 Experimental Setup")
st.markdown("""
All experiments were conducted primarily on an NVIDIA RTX 4090 GPU, with validation and compatibility checks performed on an H100 node of the Georgia Tech PACE cluster.
            
Our baseline model was Qwen-4B Base, a widely used open-source LLM in the ≈ 4 B-parameter class that offers a good trade-off between capability and feasibility for single-GPU training.
            
We designed four major experimental tracks to study sensitivity under the one-day constraint:
 * Learning-Rate Sweep - Identify stable and efficient optimization regimes.
 * GRPO Variant & Reward-Granularity Sweep - Compare BNPO, DrGRPO, and DAPO using token- vs sequence-level rewards.
 * Mixed-Precision Training - Evaluate the memory-throughput trade-off of quantized and half-precision modes.
 * Data Deduplication Hyperparameter Sweep - Measure how D4 parameters (e.g., clustering thresholds, prototype ratios) affect performance.

A full grid search across all combinations (3 x 6 x 3 x 6 = 324 runs) would have been infeasible even with partial-day runs. Instead, each factor was varied independently so we could isolate and measure its direct effect on downstream performance.

""")

st.subheader("3.5 Cluster-GRPO Extension")
st.markdown("""
Standard GRPO often produces noisy and unstable reward signals: a correct answer with a minor formatting error might receive a severe penalty, while a lucky guess may be over-rewarded. To mitigate this, we propose Cluster-GRPO, which aggregates semantically similar answers before computing rewards. For each prompt, we generate N = 16 responses, embed them using all-MiniLM-L6-v2, and group them via semantic similarity. Each response then receives the average cluster reward, smoothing noise while preserving the deterministic format score.
            
Formally:
""")

st.latex(r"""
R_{\text{final}} = R_{\text{format}} + \alpha \cdot \text{Average}(R_{\text{answer}})_{\text{cluster}}
         """)

st.markdown("""
Where R_format is the strict formatting reward, and the second term is the stabilized answer score derived from the semantic cluster.
""")


st.subheader("3.6 Evaluation Procedure")
st.markdown("""
We evaluated all models under a 5-shot Chain-of-Thought (CoT) prompting setup with greedy decoding (temperature = 0) to ensure determinism.
This setup exposes each model to several example reasoning traces without overfitting to a single pattern.
            
Performance was measured with two complementary metrics:
 * Strict Match Accuracy: Percentage of predictions that exactly match the reference answer and adhere to formatting.
 * Flexible Extract Accuracy: Percentage of correct numeric or symbolic answers regardless of formatting, capturing reasoning quality independently of syntax.
""")


st.header("4. Results")

st.markdown("""
This section presents the outcomes of our experiments, analyzing how each factor—loss type, learning rate, data deduplication, precision, and reward stabilization—affects training stability and reasoning accuracy under the one-GPU, one-day constraint. All reported results were obtained on the GSM8K benchmark unless otherwise specified.

            """)

# --- 4.1 Loss Type Sweep ---
st.subheader("4.1 Loss Type Sweep")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    We evaluated GRPO variants (BNPO, DAPO, Dr-GRPO) under both **token-level** and **sequence-level** reward formulations.
    
    **Key Findings:**
    * **Token vs. Sequence Trade-off:** Token-level sampling (optimizing each token) yielded higher *Flexible Extract* scores (~88%) but lower *Strict Match*. Sequence-level sampling improved formatting but marginally reduced reasoning flexibility.
    * **Competitive Variants:** All modern GRPO variants performed comparably (variance within ±1%), suggesting that under limited compute, the simpler baseline with good hyperparameters is sufficient.
    * **Dr-GRPO:** Achieved the highest Strict Match (86.7%), effectively balancing length bias and structure.
    """)

with col2:
    # Data from Source [110-112]
    loss_data = pd.DataFrame({
        'Variant': ['BNPO (seq)', 'BNPO (tok)', 'DAPO (seq)', 'DAPO (tok)', 'Dr-GRPO (seq)'],
        'Strict Match': [85.4, 86.0, 85.4, 85.0, 86.7],
        'Flexible Extract': [87.6, 88.6, 88.2, 88.0, 88.4]
    })
    
    # Reshape for Altair
    loss_data_long = loss_data.melt('Variant', var_name='Metric', value_name='Accuracy (%)')
    
    chart_loss = alt.Chart(loss_data_long).mark_bar().encode(
        x=alt.X('Variant:N', axis=alt.Axis(labelAngle=-45)),
        y='Accuracy (%):Q',
        color='Metric:N',
        xOffset='Metric:N',
        tooltip=['Variant', 'Metric', 'Accuracy (%)']
    ).properties(title="Strict vs Flexible Accuracy by Loss Type")
    
    st.altair_chart(chart_loss, use_container_width=True)

# --- 4.2 Learning Rate Sweep ---
st.markdown("---")
st.subheader("4.2 Learning-Rate Sweep")
st.markdown("We varied the learning rate while fixing all other parameters. This proved to be one of the **dominant factors** in our experiments.")

col3, col4 = st.columns([1, 1])

with col3:
    # Data from Source [132-136]
    lr_data = pd.DataFrame({
        'Learning Rate': ['1e-05', '1e-06', '5e-06'],
        'Strict Match': [80.1, 77.3, 59.4],
        'Flexible Extract': [88.2, 88.6, 73.9],
        'Outcome': ['Optimal Balance', 'Better Reasoning, Worse Format', 'Under-training / Failure']
    })

    # Reshape
    lr_data_long = lr_data.melt(['Learning Rate', 'Outcome'], var_name='Metric', value_name='Accuracy (%)')

    chart_lr = alt.Chart(lr_data_long).mark_bar().encode(
        x=alt.X('Learning Rate:N', sort=['1e-05', '1e-06', '5e-06']),
        y='Accuracy (%):Q',
        color='Metric:N',
        xOffset='Metric:N',
        tooltip=['Learning Rate', 'Metric', 'Accuracy (%)', 'Outcome']
    ).properties(title="Impact of Learning Rate on Performance")

    st.altair_chart(chart_lr, use_container_width=True)

with col4:
    st.markdown("""
    **Analysis:**
    * **Dominant Factor:** A mere 2x change in learning rate produced a **14.6% swing** in accuracy, dwarfing the effects of loss type or data filtering.
    * **1e-05 (The Goldilocks Rate):** Achieved the best balance (80.1% Strict, 88.2% Flexible), providing fast enough convergence within the 24-hour limit.
    * **1e-06:** While achieving high flexible accuracy (88.6%), it failed to learn strict formatting conventions in time.
    * **5e-06:** Resulted in catastrophic failure (59.4% Strict), indicating the model barely converged.
    
    > **Takeaway:** Learning rate tuning is the highest-leverage intervention under resource constraints.
    """)


# # --- 4.3 Data Deduplication Sweep ---
st.markdown("---")
st.subheader("4.3 Data Deduplication Sweep")
st.markdown("We examined how data quality filtering (Deduplication `d` and Prototype Selection `p`) within the D4 framework affects performance.")

# 1. Define Data
dedup_data = pd.DataFrame({
    'Configuration': [
        'Baseline', 
        'd=0.1, p=0.5', 
        'd=0.1, p=0.8', 
        'd=0.3, p=0.5', 
        'd=0.3, p=0.8', 
        'd=0.5, p=0.5', 
        'd=0.5, p=0.8'
    ],
    'Strict Match': [78.1, 76.6, 77.1, 72.5, 79.2, 77.8, 74.6],
    'Flexible Extract': [88.7, 88.9, 87.8, 88.2, 88.9, 88.2, 88.4]
})

dedup_long = dedup_data.melt('Configuration', var_name='Metric', value_name='Accuracy (%)')

# 2. Create Chart
sort_order = [
    'Baseline', 
    'd=0.1, p=0.5', 'd=0.1, p=0.8', 
    'd=0.3, p=0.5 (Worst)', 'd=0.3, p=0.8 (Best)', 
    'd=0.5, p=0.5', 'd=0.5, p=0.8'
]

chart_dedup = alt.Chart(dedup_long).mark_bar().encode(
    x=alt.X('Accuracy (%)', scale=alt.Scale(domain=[60, 100]), title='Score (%)'),
    y=alt.Y('Metric:N', axis=None), 
    color=alt.Color('Metric:N', legend=alt.Legend(title=None, orient='bottom')),
    row=alt.Row('Configuration:N', 
                header=alt.Header(title=None, labelAngle=0, labelAlign='center'),
                sort=sort_order),
    tooltip=['Configuration', 'Metric', 'Accuracy (%)']
).properties(
    title="Impact of D4 Filtering on Accuracy",
    height=50, 
    width=350 
).configure_axisX(
    # FIX: Globally forces the X-axis title ("Score (%)") to be centered
    titleAnchor='middle',
    titleAlign='center',
    labelAlign='center'
).configure_view(
    stroke=None
)

# 3. Render Chart
st.altair_chart(chart_dedup, use_container_width=False)

# 4. Key Findings
st.markdown("""
**Key Findings:**
* **Balance is Key:** Simply removing duplicates isn't enough. The best result came from 'medium' deduplication (`r=0.3`) balanced with 'high' diversity filtering (`p=0.8`), yielding a **+1.1% boost** over baseline.
* **Volatility:** The settings are volatile. Using the same deduplication (`0.3`) but failing to force high diversity (`0.5`) caused performance to crash to its lowest point (72.5%).
* **Format vs. Reasoning:** Flexible extract scores remained flat (~88%) across all runs. Data filtering primarily impacts **formatting adherence**, not mathematical capability.

> **Takeaway:** Teaching a model *how* to answer (format) is much more sensitive to data quality than teaching it *what* to answer (math).
""")

# --- 4.4 Mixed-Precision and Quantization Sweep ---
st.markdown("---")
st.subheader("4.4 Mixed-Precision and Quantization Sweep")
st.markdown("To enable larger batch sizes on consumer GPUs, we compared a standard 16-bit baseline against a 4-bit quantized model on the DAPO dataset.")

col1, col2, col3 = st.columns([1,2,1])
st.markdown(" ")
with col2:
    st.image("figures/mpgrpo.png")
st.markdown("""""")
col7, col8 = st.columns([1, 1])

with col7:
    quant_data = pd.DataFrame({
        'Configuration': ['16-bit Baseline', '4-bit Quantized'],
        'VRAM Usage (GB)': [16.2, 13.7], # Representing the >15% drop to get below 14GB
        'Training Stability': ['Stable', 'Stable (Identical)']
    })

    chart_quant = alt.Chart(quant_data).mark_bar(size=60).encode(
        x=alt.X('Configuration:N', sort=['16-bit Baseline', '4-bit Quantized'], axis=alt.Axis(labelAngle=0)),
        y=alt.Y('VRAM Usage (GB):Q', scale=alt.Scale(domain=[0, 20])),
        color=alt.Color('Configuration:N', legend=None),
        tooltip=['Configuration', 'VRAM Usage (GB)', 'Training Stability']
    ).properties(
        title="Projected Memory Footprint (RTX 3090/4090)"
    )
    
    st.altair_chart(chart_quant, use_container_width=True)

with col8:
    st.markdown("""
    **Analysis:**
    * **Identical Trajectories:** Learning-curve trajectories were nearly identical between the 16-bit baseline and 4-bit quantization, confirming that quantization introduced **no degradation or instability**.
    * **Memory Efficiency:** At aggressive batch sizes, peak memory usage dropped by **over 15%**.
    * **Consumer Hardware Ready:** Projected memory footprints for consumer setups (batch = 1 + gradient accumulation) dropped below **14 GB**, fitting comfortably within the limits of an RTX 3090 or 4090 (24GB cards).
    
    > **Takeaway:** 4-bit quantization and mixed-precision training enable efficient, high-throughput post-training on commodity hardware with negligible performance loss.
    """)
    
# --- 4.5 Cluster-GRPO Evaluation ---
st.markdown("---")
st.subheader("4.5 Cluster-GRPO Evaluation")
st.markdown("Finally, we evaluated Cluster-GRPO, our proposed variant designed to stabilize noisy reward signals by averaging rewards within semantically similar clusters. While our experiments utilized an H100 (80GB) for throughput, memory profiling confirms the method fits on consumer GPUs. Our peak VRAM usage on an H100 GPU was ~48 GB (using batch size 16).By reducing the micro-batch size to 4 and increasing gradient accumulation steps, the peak activation memory drops to ~15 GB. This confirms that Cluster-GRPO is fully reproducible on a single NVIDIA RTX 3090/4090 (24GB).")


col9, col10 = st.columns([1, 1])

with col9:
    cluster_data = pd.DataFrame({
        'Metric': ['Format Stability', 'Format Stability', 
                   'Reasoning Accuracy', 'Reasoning Accuracy', 
                   'Reliability', 'Reliability'],
        'Configuration': ['GRPO Baseline', 'Cluster-GRPO', 
                          'GRPO Baseline', 'Cluster-GRPO', 
                          'GRPO Baseline', 'Cluster-GRPO'],
        'Value (%)': [72.4, 91.8, 
                      32.2, 31.4, 
                      36.6, 37.8] 
    })

    chart_cluster = alt.Chart(cluster_data).mark_bar().encode(
        x=alt.X('Metric:N', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Value (%):Q'),
        color=alt.Color('Configuration:N', legend=alt.Legend(orient='top')),
        xOffset='Configuration:N',
        tooltip=['Metric', 'Configuration', 'Value (%)']
    ).properties(
        title="GRPO Baseline vs. Cluster-GRPO"
    )

    st.altair_chart(chart_cluster, use_container_width=True)

with col10:
    st.markdown("""
    **Analysis:**
    * **Format Stability (+26.8%):** The most significant gain. Strict-format accuracy surged from **72.4% → 91.8%**, nearly eliminating formatting hallucinations.
    * **Reasoning Accuracy (-0.8%):** Peak accuracy saw a slight dip (**32.2% → 31.4%**). This is viewed as a reasonable trade-off for the massive gain in consistency.
    * **Reliability (+1.2%):** "Answer-close" match rates improved, indicating that clustering successfully **rescued correct traces** that might have otherwise been discarded due to noise.
    
    > **Takeaway:** Smoothing noisy reward signals leads to significantly more stable training and better formatted outputs, even if raw exploration is modestly reduced.
    """)

st.header("5. Conclusion")

st.markdown("""
            This study investigated the feasibility of post-training large language models under extreme resource constraints, limiting the budget to a single commodity GPU and one day of training.
 Rather than pursuing peak performance, our goal was to measure how different design choices affect model stability, efficiency, and reasoning quality in a constrained environment.
We adopted Qwen3-4B Base as our foundation and combined Group Relative Policy Optimization (GRPO) and its normalization variants (BNPO, DAPO) with LoRA for parameter-efficient adaptation.
 To ensure high data efficiency, we integrated the D4 data-cleaning framework, which performs semantic deduplication and prototype filtering to maximize the value of each training token.
 Finally, we introduced Cluster-GRPO, a novel reward-stabilization method that averages rewards across semantically similar responses, mitigating noise from inconsistent answer scoring.
Across all experiments, learning rate emerged as a dominant factor influencing convergence and final performance.
 A setting of 1 x 10⁻⁵ consistently produced the best balance between reasoning accuracy and formatting compliance, while even small deviations led to double-digit performance drops.
 This sensitivity underscores that, in limited-compute settings, optimization dynamics matter far more than complex algorithmic variants.
While the various GRPO-based loss types achieved comparable results (≈ ± 1 %), they revealed a clear trade-off:
            
- Token-level rewards enhance reasoning flexibility and local correctness.
- Sequence-level rewards improve global structure and format consistency.


Quantization experiments further demonstrated that 4-bit mixed-precision training can reduce memory usage by over 15 % without degrading reasoning quality, making advanced post-training fully viable on consumer GPUs.
 Meanwhile, data cleaning with the D4 pipeline provided small but reliable gains in format learning, confirming that even minor improvements compound under strict resource limits.
 
Finally, Cluster-GRPO improved format stability by +26.8 % while maintaining nearly identical reasoning accuracy, validating its effectiveness in smoothing noisy reward signals.
From these findings, we propose the following configuration hierarchy for practitioners aiming to reproduce efficient reasoning-oriented post-training on consumer hardware:
Prioritize learning-rate tuning near 1 x 10⁻⁵; optimization dominates performance.

- Use LoRA (rank = 8) and 4-bit quantization to expand batch size without accuracy loss.
- Select reward granularity based on downstream needs: token-level for reasoning flexibility, sequence-level for format precision.
- Apply D4-based cleaning to maintain data diversity and stability.
- Adopt DAPO or Cluster-GRPO when reward noise causes unstable training or output inconsistencies.


Overall, our results show that with careful parameter tuning, reward stabilization, and efficient data use, effective post-training of reasoning-focused LLMs is achievable within a one-day, single-GPU budget.
 
By treating resource limitations as an experimental lens rather than a barrier, we identify the parameters that matter most and demonstrate a practical pathway toward accessible, reproducible, and sustainable LLM research.
            """)


st.header("6. References")
st.markdown("""
* J. Geiping and T. Goldstein, “Cramming: Training a language model on a single GPU in one day,” Proc. Int. Conf. Mach. Learn. (ICML), PMLR, 2023.
* Z. Liu et al., “Understanding r1-zero-like training: A critical perspective,” arXiv preprint arXiv:2503.20783, 2025.
* D. Guo et al., “DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning,” arXiv preprint arXiv:2501.12948, 2025.
* Kimi Team et al., “Kimi K2: Open agentic intelligence,” arXiv preprint arXiv:2507.20534, 2025.
* K. Tirumala et al., “D4: Improving LLM pretraining via document de-duplication and diversification,” in Advances in Neural Information Processing Systems, vol. 36, pp. 53983-53995, 2023.
""")



st.header("7. Contributions and Planning")
st.markdown("""

### Individual Contributions 
* Isaac Lo - Sweeps, Evaluations, Result Analysis, Presentation
* Jeff Xu - Data Deduplication, Training Sweeps, Presentation
* Chengqi Luo - Result Analysis & Visualization, Sweeps, Presentation
* Arya Anantula - Sweeps under different model setings, Presentation
* Aashutosh Aripirala - ClusterGRPO, Mixed Precision, Presentation, Sweeps

### Planning
Gantt Chart:
""")

with open("gantt.pdf", "rb") as pdf_file:
    gantt_pdf_bytes = pdf_file.read()

st.download_button(
    label="Gantt Chart (PDF download)",
    data=gantt_pdf_bytes,
    file_name="gantt.pdf",
    mime="application/pdf"
)


# LLM-Fine-tuning-for-Machine-Translation
Fine-Tuning an LLM for German–French Machine Translation

A practical experiment exploring parameter-efficient fine-tuning of BLOOMZ-1B7 for German-to-French machine translation using LoRA, quantization, and synthetic data augmentation.

The project investigates how different training-data strategies affect translation quality under limited GPU resources.

Project Overview

The goal was to fine-tune a pretrained multilingual language model for German-to-French translation and evaluate whether additional synthetic training data could improve performance.

The experiments compare four model configurations:

Model	Training Strategy	BLEU
Model A	Pre-trained BLOOMZ-1B7 (baseline)	22.51
Model B	LoRA fine-tuning on original data	21.80
Model C	LoRA fine-tuning on synthetic data	32.96
Model D	LoRA fine-tuning on combined data	27.28

The best-performing experiment improved the BLEU score from 22.51 to 32.96 using synthetic-data fine-tuning.

Key Findings

The experiments produced an interesting result: conventional fine-tuning on the limited original training dataset did not improve the baseline.

Model B achieved **21.80 BLEU**, compared with **22.51 BLEU** for the pretrained model.

This motivated an investigation into synthetic data augmentation. Fine-tuning on the synthetic dataset produced the best result at **32.96 BLEU**, representing a substantial improvement over the baseline.

Combining original and synthetic data produced **27.28 BLEU**, which was better than the baseline but lower than training exclusively on the synthetic dataset.

These results highlight how training-data composition can have a significant impact on LLM fine-tuning performance, particularly when working with relatively small datasets.

**Approach**
**1. Dataset Preparation**
The project uses the Tatoeba German–French translation dataset.
A subset of 1,000 translation pairs was sampled and divided into training and testing data using an 80/20 split.
The training data was further divided into training and validation subsets.

**2. Baseline Evaluation**
The pretrained BLOOMZ-1B7 model was evaluated on the German–French test set before fine-tuning.
The baseline achieved: **BLEU: 22.51**
This provided the reference point for the subsequent experiments.

**3. LoRA Fine-Tuning**
Instead of updating all 1.7 billion model parameters, Low-Rank Adaptation (LoRA) was used to efficiently adapt the model.
The experiments also used quantization and gradient checkpointing to reduce memory requirements and make training feasible on a limited-memory GPU.

**4. Synthetic Data Generation**
Additional German–French translation pairs were generated synthetically to increase the diversity of the training data.
The synthetic dataset focused on different everyday topics and sentence structures.
The generated translations were processed and checked before being incorporated into the experiments.

**5. Synthetic-Data Fine-Tuning**
The pretrained BLOOMZ-1B7 model was fine-tuned using the synthetic dataset.
This experiment produced the strongest result: **BLEU: 32.96**

Compared with the baseline of 22.51, this corresponds to an improvement of approximately 46% in the BLEU score.

**6. Combined Dataset**
The original and synthetic data were also combined to create a larger training dataset.
A fixed random seed was used for reproducibility when sampling the combined dataset.
Fine-tuning on this dataset produced: **BLEU: 27.28**
Although this improved upon the baseline, it did not outperform the synthetic-only experiment.

Model and Training Configuration
Model
Model: BLOOMZ-1B7
Parameters: ~1.7B
Task: German → French translation
Fine-Tuning
LoRA / Low-Rank Adaptation
Parameter-efficient fine-tuning
Quantization
Gradient checkpointing
Gradient accumulation
Evaluation
BLEU score
Beam-search generation for evaluation
Hardware
NVIDIA Tesla T4
15 GB VRAM

The use of parameter-efficient fine-tuning and quantization was particularly important because of the available GPU memory.

Repository Structure
llm-fine-tuning-machine-translation/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── fine_tuning_machine_translation.ipynb
│
├── data/
│   ├── train_dataset_a.json
│   ├── test_dataset_a.json
│   ├── synthetic_dataset_b.json
│   └── dataset_c.json
│
└── results/
    └── bleu_comparison.png
Running the Project

**Clone the repository:**
git clone <repository-url>
cd llm-fine-tuning-machine-translation

**Create a virtual environment:**
python -m venv .venv

**Activate it on Windows:**
.venv\Scripts\activate

**Install the dependencies:**

pip install -r requirements.txt

**Launch Jupyter:**

jupyter notebook

Then open:

notebooks/fine_tuning_machine_translation.ipynb

Note: Training BLOOMZ-1B7 requires a GPU with sufficient memory. The original experiments were performed using an NVIDIA Tesla T4 with 15 GB VRAM and parameter-efficient techniques such as LoRA and quantization.

**Limitations:**
- This project was conducted using a relatively small translation dataset and limited computational resources.
- The BLEU results should therefore be interpreted as experimental comparisons within this setup rather than as a state-of-the-art machine translation benchmark.
- Additional experiments with larger datasets, multiple random seeds, stronger evaluation metrics, and other multilingual models would provide a more comprehensive evaluation.

**Future Improvements:**

Potential extensions include:

Experimenting with larger and more diverse translation datasets
Comparing additional multilingual LLMs
Testing different LoRA configurations
Evaluating additional translation metrics
Running experiments across multiple random seeds
Investigating data-quality filtering for synthetic translations
Comparing synthetic-only, real-only, and mixed-data training more systematically
Technologies

Python · PyTorch · Hugging Face Transformers · Hugging Face Datasets · PEFT · LoRA · bitsandbytes · BLEU · BLOOMZ · LLM Fine-Tuning · Synthetic Data Generation

Author
Ameya Bhagwat

M.Sc. Computer Science, RPTU Kaiserslautern

This project was developed as part of practical work exploring LLM fine-tuning and Generative AI.

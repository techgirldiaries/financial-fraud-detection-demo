# 🛡️ Agentic AI Fraud Detection Demo

A modular agent system for real-time transaction fraud detection using similarity retrieval, machine learning (ML) classification and artificial intelligence (AI) reasoning.

## 🚀 Quick Start

### 1. Setup Environment (PowerShell)

```powershell
# Navigate to project directory
cd agentic_ai_fraud_detection_demo

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.\venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# Deactivate virtual environment
deactivate
```

### 2. Run the Demo

#### Option A: Streamlit Web Interface (Recommended) ⭐

```powershell
# Launch interactive web interface
streamlit run src\dashboard.py
```

Then, ctrl+click the link to view in your browser or copy the link: **<http://localhost:8502>**

**Features:**

- 🎛️ Interactive transaction selection
- 📊 Real-time fraud detection with Visualisation
- 💭 AI reasoning explanations
- 🔍 Similar fraud pattern retrieval
- 📈 PCA Visualisation of vector embeddings

#### Option B: Command-Line Interface

```powershell
# Run with default settings
python src/main.py

# Run single sample with Visualisation
python src/main.py --samples 1

# Run 5 samples without plots
python src/main.py --samples 5 --no-plot

# Run Streamlit from main.py (alternative method)
python src/main.py --interactive
```

## 📁 Project Structure

```
agentic_ai_fraud_detection_demo/
├── src/
│   ├── streamlit_app.py      # Streamlit web interface
│   ├── main.py               # CLI entry point
│   ├── data_loader.py        # Data loading & preprocessing
│   ├── model.py              # PyTorch/Sklearn ML models
│   ├── vector_store.py       # In-memory vector similarity search
│   ├── agents.py             # Multi-agent system (Retriever, Classifier, Reasoning)
│   ├── visual.py             # PCA visualisation generation
│   └── utils.py              # Debug & utility functions
├── data/
│   ├── paysim.csv            # PaySim transaction dataset (Optional)
│   └── momtsim.csv           # Synthetic mobile money dataset (Optional)
├── plots/                    # Generated visualisations
├── requirements.txt          # Python dependencies
└── README.md                 # Technical documentation
└── COLOUR_CUSTOMISATION.md   # Streamlit colour customisation
```

## 🏗️ Architecture

### Multi-Agent System

1. **🔍 Retriever Agent**
   - Performs cosine similarity search in vector store
   - Retrieves top-k similar fraud patterns
   - Returns contextual fraud examples

2. **🤖 Classifier Agent**
   - ML-based fraud probability prediction
   - Uses Scikit-learn Logistic Regression (PyTorch fallback)
   - Augments predictions with retrieved patterns

3. **💭 Reasoning Agent**
   - Combines classifier output with retrieval context
   - Generates human-readable explanations
   - Makes final fraud or legit decision with confidence

### Data Flow

```
Transaction → Retriever → Classifier → Reasoning → Decision + Explanation
                ↓              ↓            ↓
           Vector Store    ML Model    Contextual
           (Cosine Sim)   (Sklearn)    Analysis
```

## 🛠️ Tech Stack

### Core ML/Data

- **NumPy** - Numerical computations
- **Pandas** - Data manipulation & preprocessing
- **Scikit-learn** - ML models & metrics
- **PyTorch** - Deep learning (optional, fallback)

### Visualisation

- **Matplotlib** - Plot generation
- **Seaborn** - Statistical Visualisations
- **PCA** - Dimensionality reduction for vector Visualisation

### Web Interface

- **Streamlit** - Interactive web dashboard

### Dataset

- **PaySim** - Synthetic mobile money transaction dataset
- **MoMTsim** - Alternative mobile money dataset
- **Synthetic fallback** - Auto-generated if datasets unavailable

## 📊 Features

### Current Demo Features

- ✅ Multi-agent fraud detection pipeline
- ✅ Vector similarity search (cosine similarity)
- ✅ ML classification with confidence scores
- ✅ Explainable AI reasoning
- ✅ Interactive Streamlit dashboard
- ✅ CLI for batch processing
- ✅ PCA Visualisation of fraud patterns
- ✅ System fallbacks (PyTorch → Sklearn, Real data → Synthetic)

### Production Recommendations

- 🔄 Add CIS-IEEE anonymised dataset
- 🔄 Replace in-memory vector store with **FAISS**
- 🔄 Integrate **LangChain/LangGraph** for LLM-powered reasoning
- 🔄 Deploy on **AWS SageMaker**
- 🔄 Implement model monitoring & drift detection

### Future Updates

- 🔄 Add real-time streaming with **Kafka**
- 🔄 Add A/B testing framework
- 🔄 Scale with **Ray** for distributed inference

## 🔧 Configuration

### Model Settings

Edit `src/model.py` to adjust:

- Training epochs: `epochs=5`
- Batch size: `batch_size=1024`
- Learning rate: `lr=0.001`

### Agent Settings

Edit `src/agents.py` to adjust:

- Top-k retrieval: `top_k=3`
- Fraud threshold: `threshold=0.5`
- Confidence calculation method

## 🐛 Troubleshooting

### PyTorch DLL Error (Windows)

If you see `OSError: DLL initialization failed`, the system automatically falls back to Scikit-learn. To fix PyTorch:

```powershell
# Option 1: Reinstall PyTorch CPU version
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Option 2: Install Visual C++ Redistributable
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
```

### Import Errors

```powershell
# Ensure you are in the project root and venv is activated
cd agentic_ai_fraud_detection_demo
.\venv\Scripts\Activate.ps1
```

### Streamlit Not Found

```powershell
pip install streamlit
```

## 📈 Example Output

### CLI Mode

```
=== Agentic AI Demo (CLI Mode) ===

Sklearn Metrics: Acc=0.565, Prec=0.514, Rec=1.000, F1=0.679
Sample 1 (True Fraud: 0.0): Features=[ 0.1027436  -0.08124469]...
Retriever: Retrieved 3 similar patterns.
Classifier: Fraud prob = 0.088 (aug: 0.438)
Reasoning: LEGIT (conf: 0.088) - Prob: 0.088, Patterns: ['High amount transfer: 152961...']
------------------------------------------------------------
Saved Visualisation to plots\sample_0_retrieved.png
```

### Streamlit Mode

- ✅ Model trained! Dataset: 4862220 transactions, 2233935 fraud cases
- 🎛️ Interactive controls for sample selection
- 📊 Visual fraud probability metrics
- 💭 Detailed reasoning with retrieved patterns
- 📈 PCA embedding Visualisation

## 🧪 Testing

```powershell
# Run quick test with 1 sample
python src/main.py --samples 1

# Test Streamlit app
streamlit run src/dashboard.py
```

## 📝 Notes

- **Dataset**: Uses synthetic data by default. Place `paysim.csv` or `momtsim.csv` in `data/` folder for real datasets.
- **Performance**: Demo uses ~5M transactions.
- **Model**: Simple logistic regression for demo.
- **Vector Store**: In-memory for demo.

## 🤝📧 Contributing or Contact

Please open an issue on the repo to contribute, ask questions or give feedback about the demo.

## 📜 License

Licensed under the PolyForm Noncommercial License - Commercial use is prohibited

Copyright (c) 2025 Oluwakemi Obadeyi


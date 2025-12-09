"""
Streamlit Interactive Demo for Agentic AI Fraud Detection.
"""

import os
import sys
import streamlit as st

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_and_preprocess_data
from model import train_classifier
from vector_store import SimpleVectorStore
from agents import MultiAgentSystem
from visual import generate_sample_plot

# Page config
st.set_page_config(
    page_title="Agentic AI Fraud Detection Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    /* Main title styling */
    h1 {
        color: #FF4B4B;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, rgba(255,75,75,0.1) 0%, rgba(61,213,109,0.1) 100%);
        border-radius: 10px;
    }
    
    /* Fraud alert styling */
    .fraud-alert {
        background-color: rgba(255, 75, 75, 0.2);
        border-left: 5px solid #FF4B4B;
        padding: 1rem;
        border-radius: 5px;
        color: #FF4B4B;
        font-weight: bold;
    }
    
    /* Legit styling */
    .legit-alert {
        background-color: rgba(61, 213, 109, 0.2);
        border-left: 5px solid #3DD56D;
        padding: 1rem;
        border-radius: 5px;
        color: #3DD56D;
        font-weight: bold;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #262730 0%, #1a1d24 100%);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 5px;
    }
    
    /* Code blocks */
    code {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Agentic AI Fraud Detection Demo")
st.markdown("**Multi-Agent System and Retrieval Agent Reasoning for Adaptive Fraud Detection**")
st.markdown("Select a transaction sample below to run the fraud detection pipeline with Retriever, Classifier and Reasoning agents.")

# Load data and model (cached for performance)
@st.cache_data
def load_all():
    """Load and preprocess data, train model, initialise vector store and agents."""
    df, X, y = load_and_preprocess_data()
    model, fraud_vectors, fraud_texts = train_classifier(df, X, y)
    vector_store = SimpleVectorStore(fraud_vectors, fraud_texts)
    system = MultiAgentSystem(vector_store, model)
    return df, X, y, system, fraud_vectors, fraud_texts

# Load with spinner
with st.spinner("🔄 Loading data and training model... (this may take a moment)"):
    df, X, y, system, fraud_vectors, fraud_texts = load_all()

st.success(f"✅ Model trained! Dataset: {len(X)} transactions, {int(y.sum())} fraud cases")

# Sidebar controls
st.sidebar.header("🎛️ Controls")
st.sidebar.markdown("---")

# Sample selection
max_samples = min(500, len(X))
sample_idx = st.sidebar.number_input(
    "Select Sample Index",
    min_value=0,
    max_value=max_samples - 1,
    value=0,
    step=1,
    help="Choose a transaction index to analyse"
)

# Options
show_plot = st.sidebar.checkbox("📊 Generate Visualisation", value=True, help="Show PCA plot of retrieved fraud patterns")
show_details = st.sidebar.checkbox("🔍 Show Detailed Features", value=False, help="Display all transaction features")

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run Fraud Detection", type="primary", use_container_width=True)

# Main content area
if run_button:
    with st.spinner("🔍 Analysing transaction..."):
        # Get transaction data
        tx_features = X[sample_idx:sample_idx+1].squeeze()
        tx_desc = str(df.iloc[sample_idx].get('type', 'Unknown'))
        is_true_fraud = float(y[sample_idx])
        
        # Run detection
        result = system.detect_fraud(tx_features, tx_desc)
        st.write("DEBUG - Raw result:", result)  # Add this line temporarily
        
        # Display results in columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader(f"📋 Transaction #{sample_idx}")
            
            # Transaction info with color-coded badges
            if is_true_fraud == 1:
                st.markdown("""
                    <span style="background-color: rgba(255,75,75,0.3); 
                                padding: 0.3rem 0.8rem; 
                                border-radius: 15px; 
                                color: #FF4B4B;
                                font-weight: bold;">
                        🚨 Ground Truth: FRAUD
                    </span>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <span style="background-color: rgba(61,213,109,0.3); 
                                padding: 0.3rem 0.8rem; 
                                border-radius: 15px; 
                                color: #3DD56D;
                                font-weight: bold;">
                        ✅ Ground Truth: LEGIT
                    </span>
                """, unsafe_allow_html=True)
            
            st.markdown(f"**Transaction Type:** `{tx_desc}`")
            
            # Features
            if show_details:
                st.markdown("**All Features:**")
                features_dict = {f"feature_{i}": float(val) for i, val in enumerate(tx_features)}
                st.json(features_dict)
            else:
                st.markdown("**Sample Features (first 5):**")
                st.code(f"{tx_features[:5]}")
        
        with col2:
            st.subheader("🤖 Detection Result")
            
            # Decision badge with custom styling
            decision = result.get('decision', 'UNKNOWN')
            confidence = result.get('confidence', 0.0)
            
            if decision == 'FRAUD':
                st.markdown(f"""
                    <div class="fraud-alert">
                        🚨 <span style="font-size: 1.5rem;">FRAUD DETECTED</span><br>
                        Confidence: {confidence:.1%}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="legit-alert">
                        ✅ <span style="font-size: 1.5rem;">LEGITIMATE</span><br>
                        Confidence: {confidence:.1%}
                    </div>
                """, unsafe_allow_html=True)
            
            # Key metrics
            fraud_prob = result.get('fraud_prob', 0.0)
            st.metric("Fraud Probability", f"{fraud_prob:.3f}", delta=f"{fraud_prob - 0.5:.3f} vs threshold")
            
            # Reasoning
            st.markdown("**💭 Reasoning:**")
            reasoning = result.get('reasoning', 'No reasoning available')
            st.info(reasoning)
        
        # Retrieved patterns
        st.markdown("---")
        st.subheader("🔎 Retrieved Similar Fraud Patterns")
        retrieved = result.get('retrieved', [])
        if retrieved:
            for idx, pattern in enumerate(retrieved[:5], 1):
                with st.expander(f"Pattern {idx}"):
                    st.text(pattern)
        else:
            st.warning("No similar patterns retrieved")
        
        # Visualisation
        if show_plot:
            st.markdown("---")
            st.subheader("📊 PCA Visualisation: Query vs Retrieved Patterns")
            with st.spinner("Generating visualisation..."):
                plot_path = generate_sample_plot(sample_idx, X, fraud_vectors, fraud_texts)
                if plot_path and os.path.exists(plot_path):
                    st.image(plot_path, use_container_width=True)
                    st.caption(f"PCA projection showing the query transaction (red star) and top 5 retrieved fraud patterns (blue dots)")
                else:
                    st.warning("⚠️ Visualisation not available (matplotlib may not be installed)")
        
        # Full result details
        with st.expander("🔧 Full Technical Details"):
            st.json(result)

else:
    # Instructions when no detection has been run
    st.info("👈 Use the sidebar to select a transaction and click **Run Fraud Detection** to begin")
    
    # Show sample statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", len(X))
    with col2:
        st.metric("Fraud Cases", int(y.sum()))
    with col3:
        fraud_rate = (y.sum() / len(y)) * 100
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

# Footer with demo info
st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ About This Demo App"):
    st.markdown("""
    **Multi-Agent Architecture:**
    - 🔍 **Retriever Agent**: Finds similar fraud patterns using vector similarity
    - 🤖 **Classifier Agent**: ML-based fraud probability prediction
    - 💭 **Reasoning Agent**: Combines signals and generates explanation
    
    **Technology Stack:**
    - Data: PaySim synthetic financial dataset
    - Model: Scikit-learn Logistic Regression (PyTorch fallback available)
    - Vector Store: In-memory cosine similarity search
    - Visualisation: PCA projection with matplotlib
    
    **For Production:**
    - Replace with FAISS for scalable vector search
    - Use LangChain for LLM-powered reasoning
    - Deploy on AWS SageMaker Studio Lab
    """)

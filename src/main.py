"""
Entry point for the Agentic AI Fraud Detection Demo.

Supports CLI mode (default) and Streamlit interactive mode.
"""

import argparse
import os
import sys
from typing import Any

# Local imports
from data_loader import load_and_preprocess_data
from model import train_classifier
from vector_store import SimpleVectorStore
from agents import MultiAgentSystem
from visual import generate_sample_plot
import utils

def run_cli(no_plot: bool = False, samples: int = 3):
    print('\n=== Agentic AI Demo (CLI Mode) ===\n')
    utils.check_windows_runtime_samples()

    df, X, y = load_and_preprocess_data()
    model, fraud_vectors, fraud_texts = train_classifier(df, X, y)
    vector_store = SimpleVectorStore(fraud_vectors, fraud_texts)
    system = MultiAgentSystem(vector_store, model)

    sample_indices = [0, 100, 500][:samples]
    for i in sample_indices:
        if i >= len(X):
            break
        tx_features = X[i:i+1].squeeze()
        tx_desc = str(df.iloc[i].get('type', ''))
        is_true_fraud = float(y[i])
        print(f"Sample {i+1} (True Fraud: {is_true_fraud}): Features={tx_features[:2]}...")
        result = system.detect_fraud(tx_features, tx_desc)
        filtered = {k: v for k, v in result.items() if k in ['decision', 'confidence', 'reasoning']}
        print(result['reasoning'])  # Simplified output for CLI
        print('-' * 60)

        if not no_plot:
            path = generate_sample_plot(i, X, fraud_vectors, fraud_texts)
            if path:
                print(f'Saved visualisation to {path}')

    print('\nDemo finished.')

def run_interactive():
    """Launch Streamlit demo."""
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not installed. Run: pip install streamlit")
        sys.exit(1)

    # Inline Streamlit code (or import from demo_app.py if separate)
    st.set_page_config(page_title="Agentic AI Fraud Detection Demo", layout="wide")
    st.title("🛡️ Agentic AI Fraud Detection Demo")
    st.markdown("Select a transaction sample below to run the multi-agent and retrieval-based fraud detection pipeline.")

    # Load data and model (use st.cache_data if in separate file)
    @st.cache_data
    def load_all():
        df, X, y = load_and_preprocess_data()
        model, fraud_vectors, fraud_texts = train_classifier(df, X, y)
        vector_store = SimpleVectorStore(fraud_vectors, fraud_texts)
        system = MultiAgentSystem(vector_store, model)
        return df, X, y, system

    with st.spinner("Loading data and training model..."):
        df, X, y, system = load_all()

    # Sidebar
    st.sidebar.header("Controls")
    sample_idx = st.sidebar.selectbox("Select Sample Index", options=list(range(min(100, len(X)))), index=0)
    show_plot = st.sidebar.checkbox("Generate & Show Plot", value=True)

    if st.sidebar.button("Run Detection"):
        tx_features = X[sample_idx:sample_idx+1].squeeze()
        tx_desc = str(df.iloc[sample_idx].get('type', ''))
        is_true_fraud = float(y[sample_idx])
        result = system.detect_fraud(tx_features, tx_desc)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Sample {sample_idx} (True Fraud: {is_true_fraud})")
            st.write("**Transaction Features:**")
            st.json({f"feature_{i}": float(val) for i, val in enumerate(tx_features)})
            st.write("**Description:**", tx_desc)

        with col2:
            filtered = {k: v for k, v in result.items() if k in ['decision', 'confidence', 'reasoning', 'retrieved', 'fraud_probability']}
            st.subheader("Detection Result")
            st.json(filtered)

        if show_plot:
            with st.spinner("Generating plot..."):
                path = generate_sample_plot(sample_idx, X, system.agents[0].vector_store.fraud_vectors, 
                                        system.agents[0].vector_store.fraud_texts)
                if path and os.path.exists(path):
                    st.image(path, caption=f"PCA Visualisation: Top 5 Retrieved Fraud Patterns", use_container_width=True)
                else:
                    st.warning("Plot generation failed or unavailable.")

    with st.expander("Demo App Overview"):
        st.markdown("""
        - **Agents**: Retriever (similarity search), Classifier (ML prediction + augmentation), Reasoning (decision & explanation).
        - **Data**: Synthetic transactions.
        - **Model**: PyTorch NN or Sklearn fallback.
        - **Scalability**: In-memory for demo; production: FAISS/LangChain/AWS.
        """)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fraud Detection Demo Runner")
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('--samples', type=int, default=3, help='Number of samples (CLI mode)')
    parser.add_argument('--interactive', action='store_true', help='Run Streamlit interactive demo')
    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    else:
        run_cli(no_plot=args.no_plot, samples=args.samples)
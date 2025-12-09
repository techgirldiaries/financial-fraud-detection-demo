"""
Agent classes for the multi-agent fraud detection system.
"""

import numpy as np
from typing import Any, Dict, List, Optional
from vector_store import SimpleVectorStore

class Agent:
    def __init__(self, name: str, vector_store: Optional[SimpleVectorStore] = None, model: Optional[Any] = None):
        self.name = name
        self.vector_store = vector_store
        self.model = model

    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class RetrieverAgent(Agent):
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        tx_features = message['features']
        assert self.vector_store is not None
        retrieved = self.vector_store.retrieve(tx_features)
        response = {"agent": self.name, "retrieved": retrieved, **message}
        print(f"{self.name}: Retrieved {len(retrieved)} similar patterns.")
        return response

class ClassifierAgent(Agent):
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        tx_features = message['features']
        assert self.model is not None

        base_probability = 0.0
        # Torch inference
        if hasattr(self.model, 'eval'):  # PyTorch check
            import torch
            with torch.no_grad():
                inp = torch.from_numpy(np.asarray(tx_features)).unsqueeze(0).float()
                out = self.model(inp).squeeze()
                base_probability = float(out.item())
        else:  # Sklearn
            X_in = np.asarray(tx_features).reshape(1, -1)
            if hasattr(self.model, 'predict_probability'):
                probs = self.model.predict_probability(X_in)
                base_probability = float(probs[0, 1] if probs.shape[1] > 1 else probs[0, 0])
            else:
                pred = self.model.predict(X_in)[0]
                base_probability = float(pred)

        retrieved = message.get('retrieved', [])
        average_similarity = np.mean([r['similarity'] for r in retrieved]) if retrieved else 0.0
        augmented_probability = base_probability + (average_similarity * 0.2)
        response = {
            "agent": self.name,
            "fraud_probability": min(1.0, float(augmented_probability)),
            "base_probability": float(base_probability),
            "augmentation": float(average_similarity),
            **message
        }
        print(f"{self.name}: Fraud probability = {response['fraud_probability']:.3f} (aug: {average_similarity:.3f})")
        return response

class ReasoningAgent(Agent):
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        prob = message['fraud_probability']
        retrieved = message.get('retrieved', [])
        decision = "FRAUD" if prob > 0.7 else "SUSPICIOUS" if prob > 0.3 else "LEGIT"
        confidence = float(prob)
        retrieved_short = [r.get('pattern','')[:50] + '...' for r in retrieved[:2]]
        reasoning = f"Prob: {prob:.3f}, Patterns: {retrieved_short}"
        response = {
            "agent": self.name,
            "decision": decision,
            "confidence": confidence,
            "reasoning": reasoning,
            **message
        }
        print(f"{self.name}: {decision} (conf: {confidence:.3f}) - {reasoning[:100]}")
        return response

class MultiAgentSystem:
    def __init__(self, vector_store: SimpleVectorStore, model: Any):
        self.agents = [
            RetrieverAgent("Retriever", vector_store),
            ClassifierAgent("Classifier", vector_store, model),
            ReasoningAgent("Reasoning", vector_store, model)
        ]

    def detect_fraud(self, tx_features: np.ndarray, tx_desc: str = "") -> Dict[str, Any]:
        message = {"features": tx_features, "desc": tx_desc}
        for agent in self.agents:
            message = agent.process(message)
        return message
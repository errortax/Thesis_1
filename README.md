review of the churn models : https://www.preprints.org/manuscript/202503.1969/v2
Challenges and Limitations:
Class Imbalance: Fewer churners than non-churners in datasets bias models toward the majority class. Resampling and cost-sensitive learning help but may cause overfitting or higher computational costs.

Feature Engineering & Data Representation: Integrating diverse data sources (call logs, social media, support interactions) is complex. DL automates feature extraction but needs heavy preprocessing and resources.

Model Interpretability: DL models (e.g., neural networks) are black-box, unlike interpretable models like DTs or logistic regression. Explainable AI (SHAP, attention mechanisms) exists but lacks widespread adoption.

Dynamic Customer Behavior: Concept drift (changing preferences/engagement) challenges static models. Adaptive learning (online/reinforcement learning) helps but requires continuous retraining.

Evaluation Metrics vs. Business Impact: Accuracy, F1-score, and AUC-ROC don’t always align with business needs. Profit-driven metrics (cost of retention vs. lost revenue) are underexplored.

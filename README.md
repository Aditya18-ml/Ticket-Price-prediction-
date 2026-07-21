# Support Ticket Prediction System

A machine learning pipeline built with **LightGBM** to automate customer support ticket classification and downtime estimation.

---

## 📌 Core Prediction Targets
* **Ticket Priority (`priority_cat`):** Multiclass model to automatically triage issue urgency (Low, Medium, High, Urgent).
* **Customer Sentiment (`customer_sentiment_cat`):** Multiclass model to predict customer satisfaction/frustration levels.
* **Downtime Duration (`downtime_min`):** Tweedie regression model to estimate system downtime in minutes.
* **Payment Impact (`payment_impact_flag`):** Weighted binary model (`scale_pos_weight=80`) to detect critical financial blockers.

---



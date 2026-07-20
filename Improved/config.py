DATA_PATH = "Support_tickets.csv" 
MODEL_DIR = "models"
OUTPUT_DATA_PATH = "Support_tickets_predictions.csv"

PIPELINE_CONFIG = {
    'sentiment': {
        'target': 'customer_sentiment_cat',
        'task_type': 'classification',
        'features': [
            'company_size_cat', 'customer_tier_cat', 'org_users', 'past_30d_tickets', 
            'past_90d_incidents', 'customers_affected', 'error_rate_pct', 
            'downtime_min', 'description_length'
        ],
        'params': {
            'learning_rate': 0.05, 'max_depth': 3, 'min_child_samples': 100, 
            'n_estimators': 50, 'num_leaves': 7, 'class_weight': 'balanced', 
            'objective': 'multiclass', 'random_state': 42, 'n_jobs': -1
        }
    },
    'priority': {
        'target': 'priority_cat',
        'task_type': 'classification',
        'features': [
            'company_size_cat', 'customer_tier_cat', 'org_users', 'past_30d_tickets', 
            'past_90d_incidents', 'customers_affected', 'error_rate_pct', 
            'downtime_min', 'description_length'
        ],
        'params': {
            'class_weight': 'balanced', 'objective': 'multiclass', 
            'random_state': 42, 'n_jobs': -1
        }
    },
    'downtime': {
        'target': 'downtime_min',
        'task_type': 'regression',
        'features': [
            'customers_affected', 'error_rate_pct', 'past_90d_incidents', 
            'security_incident_flag', 'data_loss_flag', 'payment_impact_flag', 
            'product_area_cat', 'customer_tier_cat', 'has_runbook'
        ],
        'params': {
            'objective': 'tweedie', 'tweedie_variance_power': 1.5, 
            'random_state': 42, 'n_jobs': -1
        }
    },
    'payment_impact': {
        'target': 'payment_impact_flag',
        'task_type': 'classification',
        'features': [
            'customers_affected', 'org_users', 'error_rate_pct', 'past_30d_tickets', 
            'past_90d_incidents', 'security_incident_flag', 'data_loss_flag', 
            'downtime_min', 'product_area_cat', 'customer_tier_cat', 'has_runbook', 
            'priority_cat', 'customer_sentiment_cat'
        ],
        'params': {
            'scale_pos_weight': 80, 'num_leaves': 15, 'n_estimators': 500, 
            'min_child_samples': 20, 'max_depth': 7, 'learning_rate': 0.05, 
            'random_state': 42, 'n_jobs': -1
        }
    }
}
# Automated Machine Learning Application

A complete end-to-end machine learning platform that allows non-technical users to upload datasets, select ML tasks, and receive trained models without writing any code.

## Project Structure

```
Automated-ML-Application/
├── backend/                    # FastAPI backend
│   ├── api/                   # API routes and schemas
│   ├── models/                # Saved trained models
│   ├── logs/                  # Application logs
│   ├── app.py                # FastAPI application
│   └── config.py             # Configuration settings
│
├── preprocessing/            # Data preprocessing pipeline
│   ├── data_loader.py        # Load CSV/XLSX files
│   ├── data_cleaner.py       # Handle missing values
│   ├── encoder.py            # Encode categorical variables
│   ├── scaler.py             # Scale numerical features
│   ├── balancer.py           # Handle imbalanced data
│   └── pipeline.py           # Main pipeline orchestrator
│
├── classification/           # Classification models
│   ├── models.py            # Classification algorithms
│   ├── trainer.py           # Training logic
│   └── evaluator.py         # Evaluation metrics
│
├── regression/              # Regression models
│   ├── models.py           # Regression algorithms
│   ├── trainer.py          # Training logic
│   └── evaluator.py        # Evaluation metrics
│
├── clustering/             # Clustering models
│   ├── models.py          # Clustering algorithms
│   ├── trainer.py         # Training logic
│   └── evaluator.py       # Evaluation metrics
│
├── frontend/              # Web UI
│   ├── index.html        # Main page
│   ├── css/
│   │   └── styles.css    # Styling
│   └── js/               # JavaScript logic
│
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Features

### Data Ingestion

- Upload CSV or XLSX files
- Data preview functionality
- Automatic column detection

### ML Task Selection

- Classification
- Regression
- Clustering

### Automated Data Pipeline

1. **Handle Missing Values** - Mean imputation strategy
2. **Encode Categorical Variables** - Label encoding
3. **Scale Numerical Features** - Standard scaling
4. **Balance Imbalanced Data** - Oversampling for classification

### Model Training

- Multiple algorithms per task type
- Automatic best model selection based on performance
- 80/20 train-test split

### Model Evaluation

- **Classification**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Regression**: MAE, MSE, RMSE, R² Score
- **Clustering**: Silhouette Score

### Model Export

- Download trained models as .pkl files
- Serialize preprocessing pipeline with models

## Installation

1. **Clone the repository**

   ```bash
   cd "Automated ML Application"
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Start the Backend

```bash
python backend/app.py
```

The API will be available at `http://127.0.0.1:8000`

- Interactive API docs: `http://127.0.0.1:8000/docs`
- Alternative docs: `http://127.0.0.1:8000/redoc`

### Start the Frontend

Open `frontend/index.html` in a web browser (or use a local server)

```bash
# Using Python's built-in server
python -m http.server 8080 --directory frontend
```

Then navigate to `http://127.0.0.1:8080`

## API Endpoints

| Endpoint                   | Method | Description            |
| -------------------------- | ------ | ---------------------- |
| `/api/upload`              | POST   | Upload dataset         |
| `/api/train`               | POST   | Start model training   |
| `/api/results/{task_id}`   | GET    | Get training results   |
| `/api/download/{model_id}` | GET    | Download trained model |

## Supported Algorithms

### Classification

- Logistic Regression
- Random Forest
- Support Vector Machine

### Regression

- Linear Regression
- Random Forest
- Support Vector Regression

### Clustering

- K-Means
- DBSCAN
- Hierarchical Clustering

## Team Workflow

Each team member should focus on their assigned module:

- **Backend API**: Routes, request handling, orchestration
- **Preprocessing**: Data loading, cleaning, encoding, scaling, balancing
- **Classification**: Classification models and evaluation
- **Regression**: Regression models and evaluation
- **Clustering**: Clustering models and evaluation
- **Frontend**: UI design, user interaction, visualization

## Requirements

- Python 3.8+
- All dependencies listed in `requirements.txt`

## Notes

- Maximum file size: 50 MB
- Test size: 20% by default
- Random state: 42 (for reproducibility)

## License

Project for Cairo University - Faculty of FCAI

## Deadline

**Saturday, 2nd May 2026 at 11:59 PM**

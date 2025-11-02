# Student Performance Prediction Project

## Project Overview

This project was undertaken during the **CipherSchools ML & DS Internship** and aims to predict student academic performance based on historical data including attendance, assignment scores, and test results. The project provides a comprehensive end-to-end machine learning workflow from data preprocessing to model evaluation and visualization.

## Objectives

- **Predict student performance** using historical academic data
- **Identify key factors** affecting student grades
- **Implement and evaluate** machine learning models:
  - Linear Regression (for grade prediction)
  - K-Means Clustering (for grouping students)
- **Enhance practical skills** in data preprocessing, visualization, and model evaluation
- **Derive actionable insights** from the data to improve academic outcomes

## Scope of Work

1. **Data Collection and Preprocessing**
   - Handling missing values
   - Detecting and handling outliers
   - Feature scaling and encoding

2. **Exploratory Data Analysis (EDA)**
   - Statistical summaries
   - Correlation analysis
   - Visualizations of trends and patterns

3. **Machine Learning Implementation**
   - **Supervised Learning**: Linear Regression for grade prediction
   - **Unsupervised Learning**: K-Means clustering for student grouping

4. **Model Evaluation**
   - R² score and RMSE for regression model
   - Cluster analysis and interpretation

5. **Visualization and Reporting**
   - Comprehensive visualizations
   - Summary reports with actionable insights

## Project Structure

```
Summer-Training-Project/
│
├── data/
│   └── student_performance.csv          # Generated dataset
│
├── results/                              # All outputs
│   ├── *.png                            # Visualizations
│   ├── clustered_students.csv           # Students with cluster labels
│   └── summary_report.txt               # Analysis summary
│
├── models/                               # Saved ML models
│   ├── linear_regression_model.pkl
│   └── kmeans_model.pkl
│
├── generate_data.py                     # Dataset generation
├── data_preprocessing.py                # Preprocessing pipeline
├── eda.py                               # Exploratory Data Analysis
├── models.py                            # ML models implementation
├── visualizations.py                    # Visualization functions
├── app.py                               # 🆕 Streamlit web application
├── main.py                              # Command-line execution script
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Setup

1. **Clone or download the project**

2. **Create and activate virtual environment** (Recommended):
   
   **On Windows (PowerShell):**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   
   **On Windows (Command Prompt):**
   ```cmd
   python -m venv venv
   venv\Scripts\activate.bat
   ```
   
   **On macOS/Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** If you prefer not to use a virtual environment, you can install directly, but using a virtual environment is strongly recommended to avoid conflicts with other projects.

## Usage

### 🚀 Streamlit Web Application

The project includes a **professional Streamlit dashboard** for predictions and analytics.

1. **Install dependencies** (if not already done):
   ```bash
   pip install -r requirements.txt
   ```

2. **Pre-train models** (optional - models will train automatically on first run):
   ```bash
   python train_models.py
   ```

3. **Launch the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

4. **Access the application**:
   - The app will automatically open in your default web browser
   - If not, navigate to `http://localhost:8501`

#### Dashboard Features:
- **Overview**: Dataset statistics and quick insights
- **Predict Performance**: Input student data to predict final grades
- **Analytics**: Interactive visualizations and correlation analysis
- **Insights**: Key findings, model performance, and recommendations

**Note**: Models are automatically trained and cached on first run. The training process includes progress indicators and takes a few seconds.

### 📜 Command Line Usage

You can also run the complete workflow via command line:

```bash
python main.py
```

This will:
1. Generate the synthetic student dataset (if not exists)
2. Preprocess the data
3. Perform EDA and generate visualizations
4. Train Linear Regression model
5. Train K-Means clustering model
6. Evaluate models and generate results
7. Save all outputs to respective directories

### Individual Module Execution

#### Generate Dataset Only
```bash
python generate_data.py
```

#### Run EDA Only
```bash
python eda.py data/student_performance.csv
```

#### Custom Usage
You can also import and use individual modules in your own scripts:

```python
from data_preprocessing import DataPreprocessor
from models import StudentPerformanceModels
from eda import StudentPerformanceEDA

# Your custom code here
```

## Dataset Description

The synthetic dataset includes the following features:

- **Student_ID**: Unique identifier
- **Age**: Student age
- **Gender**: Student gender (Male/Female)
- **Attendance**: Attendance percentage (0-100%)
- **Assignment_1 to Assignment_4**: Individual assignment scores (0-100)
- **Test_1, Test_2**: Quiz/test scores (0-100)
- **Midterm**: Midterm exam score (0-100)
- **Final_Exam**: Final exam score (0-100)
- **Study_Hours_Per_Week**: Hours spent studying per week
- **Previous_GPA**: Previous semester GPA (0-4.0)
- **Avg_Assignment**: Average of all assignments
- **Avg_Test**: Average of all tests
- **Final_Grade**: Calculated final grade (weighted average)
- **Grade_Letter**: Letter grade (A, B, C, D, F)

## Key Features

### Data Preprocessing
- Missing value imputation (mean, median, mode)
- Outlier detection and handling (IQR method, capping)
- Feature scaling (StandardScaler)
- Categorical encoding (LabelEncoder)

### Exploratory Data Analysis
- Distribution plots
- Correlation heatmaps
- Feature vs target relationships
- Boxplots for outlier visualization
- Statistical summaries

### Machine Learning Models

#### Linear Regression
- Predicts student final grades
- Evaluates using R² score and RMSE
- Provides feature importance (coefficients)

#### K-Means Clustering
- Groups students into performance-based clusters
- Uses elbow method to find optimal number of clusters
- Analyzes cluster characteristics

### Visualizations
- Regression results (predicted vs actual, residuals)
- Feature importance plots
- Elbow method for cluster selection
- Cluster visualizations (PCA-based)
- Cluster comparison analysis
- Grade distribution by cluster

## Results

The project generates comprehensive outputs:

1. **Visualizations** (in `results/` directory):
   - Distribution plots
   - Correlation heatmaps
   - Regression analysis plots
   - Cluster visualizations
   - Feature importance charts

2. **Models** (in `models/` directory):
   - Trained Linear Regression model
   - Trained K-Means clustering model

3. **Reports**:
   - Summary report with key findings
   - Clustered student data with labels

## Model Evaluation Metrics

- **R² Score**: Coefficient of determination (higher is better, max=1.0)
- **RMSE**: Root Mean Squared Error (lower is better)
- **Cluster Inertia**: Within-cluster sum of squares (lower is better)

## Insights and Applications

### Key Factors Affecting Performance
The model identifies the most influential factors:
- Assignment scores
- Test performance
- Attendance
- Study hours
- Previous academic performance

### Student Clustering
Students are grouped into clusters based on:
- Overall performance patterns
- Study habits
- Attendance consistency
- Test vs assignment performance

### Actionable Recommendations
1. **Identify at-risk students** early through clustering
2. **Focus interventions** on key factors (attendance, study hours)
3. **Personalize support** based on cluster characteristics
4. **Monitor progress** using predictive models

## Applications

This project demonstrates practical application in:
- **Educational Institutions**: Identify students needing support
- **Academic Analytics**: Understand performance drivers
- **Personalized Learning**: Develop targeted interventions
- **Employee Performance**: Adaptable to workforce analytics
- **Customer Segmentation**: Similar clustering approaches

## Role and Profile

**Role**: ML & DS Intern (Hybrid Mode)

**Responsibilities**:
- Collected, cleaned, and preprocessed the dataset
- Performed exploratory data analysis and visualized key trends
- Implemented Linear Regression and K-Means clustering
- Evaluated model performance and interpreted results
- Generated actionable insights for academic improvement

## Technologies Used

- **Python 3.x**
- **Streamlit**: Interactive web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning models
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical visualization
- **Plotly**: Interactive visualizations

## Future Enhancements

Potential improvements and extensions:
- Implement additional ML models (Random Forest, XGBoost, Neural Networks)
- Add real-time prediction API
- Develop interactive dashboard
- Expand dataset with more features
- Implement time series analysis for longitudinal studies
- Add ensemble methods for better predictions

## License

This project is created for educational and portfolio purposes.

## Contact

For questions or feedback about this project, please refer to the project repository.

---

**Note**: This project uses synthetic data for demonstration purposes. In a real-world scenario, proper data privacy and ethical considerations must be addressed when working with student data.


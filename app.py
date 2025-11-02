"""
Student Performance Prediction Dashboard
ML & DS Internship - CipherSchools
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
import time

# Import custom modules
from generate_data import generate_student_data
from data_preprocessing import DataPreprocessor
from models import StudentPerformanceModels

# Page configuration
st.set_page_config(
    page_title="Academic Performance Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        padding: 1.5rem 0;
        border-bottom: 3px solid #3498db;
        margin-bottom: 1rem;
        text-align: center;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-top: -1rem;
        margin-bottom: 2.5rem;
        font-style: italic;
        font-weight: 300;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stProgress > div > div > div {
        background-color: #3498db;
    }
    /* Improve sidebar */
    .css-1d391kg {
        padding-top: 2rem;
    }
    /* Better button styling */
    .stButton > button {
        border-radius: 6px;
        border: none;
        background: linear-gradient(90deg, #3498db 0%, #2980b9 100%);
        transition: all 0.3s ease;
        font-weight: 500;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.3);
    }
    /* Improve expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #2c3e50;
    }
    /* Better selectbox */
    .stSelectbox label {
        font-weight: 500;
        color: #2c3e50;
    }
    /* Improve metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    /* Better card sections */
    .element-container {
        padding: 0.5rem 0;
    }
    /* Smooth transitions */
    * {
        transition: background-color 0.2s ease;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'lr_model' not in st.session_state:
    st.session_state.lr_model = None
if 'kmeans_model' not in st.session_state:
    st.session_state.kmeans_model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'cluster_labels' not in st.session_state:
    st.session_state.cluster_labels = None

def load_or_generate_data():
    """Load existing data or generate new dataset."""
    data_path = 'data/student_performance.csv'
    
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        df = generate_student_data(n_students=500, seed=42)
        os.makedirs('data', exist_ok=True)
        df.to_csv(data_path, index=False)
        return df

def load_or_train_models(df):
    """Load pre-trained models or train if they don't exist."""
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    lr_path = f'{model_dir}/linear_regression_model.pkl'
    kmeans_path = f'{model_dir}/kmeans_model.pkl'
    preprocessor_path = f'{model_dir}/preprocessor.pkl'
    
    # Check if models exist
    if os.path.exists(lr_path) and os.path.exists(kmeans_path):
        # Load existing models
        lr_model = joblib.load(lr_path)
        kmeans_model = joblib.load(kmeans_path)
        preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None
        
        # Load processed data
        X, y, feature_names = preprocess_data(df, preprocessor)
        cluster_labels = kmeans_model.predict(X)
        
        return lr_model, kmeans_model, preprocessor, X, y, cluster_labels
    else:
        # Train new models
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Initializing models...")
        progress_bar.progress(10)
        
        # Preprocess data
        status_text.text("Preprocessing data...")
        progress_bar.progress(20)
        preprocessor = DataPreprocessor()
        X, y, feature_names = preprocessor.preprocess(
            df,
            target_column='Final_Grade',
            handle_missing=True,
            handle_outliers=True,
            encode_categorical=True,
            scale_features=True
        )
        progress_bar.progress(40)
        
        # Train Linear Regression
        status_text.text("Training Linear Regression model...")
        progress_bar.progress(50)
        models = StudentPerformanceModels(X, y, feature_names)
        models.split_data()
        lr_model = LinearRegression()
        lr_model.fit(models.X_train, models.y_train)
        progress_bar.progress(70)
        
        # Train K-Means
        status_text.text("Training K-Means clustering model...")
        progress_bar.progress(80)
        kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_labels = kmeans_model.fit_predict(X)
        progress_bar.progress(90)
        
        # Save models
        status_text.text("Saving models...")
        joblib.dump(lr_model, lr_path)
        joblib.dump(kmeans_model, kmeans_path)
        joblib.dump(preprocessor, preprocessor_path)
        progress_bar.progress(100)
        
        status_text.text("✓ Models ready")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        return lr_model, kmeans_model, preprocessor, X, y, cluster_labels

def preprocess_data(df, preprocessor=None):
    """Preprocess data using existing or new preprocessor."""
    if preprocessor is None:
        preprocessor = DataPreprocessor()
    
    X, y, feature_names = preprocessor.preprocess(
        df,
        target_column='Final_Grade',
        handle_missing=True,
        handle_outliers=True,
        encode_categorical=True,
        scale_features=True
    )
    
    return X, y, feature_names

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown(
        '<div class="main-header">🎓 Academic Performance Analytics</div>'
        '<div class="sub-header">Predictive Analytics for Student Success</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar navigation
    st.sidebar.markdown("### 🧭 Navigation")
    st.sidebar.markdown("---")
    page = st.sidebar.selectbox(
        "Select a page",
        ["Overview", "Predict Performance", "Analytics", "Insights"],
        label_visibility="visible"
    )
    
    # Initialize data and models
    if st.session_state.df is None:
        with st.spinner("Loading dataset..."):
            st.session_state.df = load_or_generate_data()
    
    if not st.session_state.models_loaded:
        with st.spinner("Loading models..."):
            lr_model, kmeans_model, preprocessor, X, y, cluster_labels = load_or_train_models(st.session_state.df)
            st.session_state.lr_model = lr_model
            st.session_state.kmeans_model = kmeans_model
            st.session_state.preprocessor = preprocessor
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.cluster_labels = cluster_labels
            st.session_state.models_loaded = True
            st.rerun()
    
    df = st.session_state.df
    
    # Overview Page
    if page == "Overview":
        st.subheader("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Students", len(df))
        with col2:
            st.metric("Average Grade", f"{df['Final_Grade'].mean():.1f}")
        with col3:
            st.metric("Avg Attendance", f"{df['Attendance'].mean():.1f}%")
        with col4:
            st.metric("Avg Study Hours", f"{df['Study_Hours_Per_Week'].mean():.1f}")
        
        st.markdown("---")
        
        # Dataset preview
        with st.expander("📋 View Dataset", expanded=False):
            st.dataframe(df.head(100), use_container_width=True, height=400)
        
        st.markdown("### Quick Statistics")
        # Quick statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Grade Distribution")
            grade_counts = df['Grade_Letter'].value_counts().sort_index()
            fig = px.bar(
                x=grade_counts.index,
                y=grade_counts.values,
                labels={'x': 'Grade', 'y': 'Number of Students'},
                color=grade_counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Performance Distribution")
            fig = px.histogram(
                df,
                x='Final_Grade',
                nbins=30,
                labels={'Final_Grade': 'Final Grade', 'count': 'Frequency'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Predict Page
    elif page == "Predict Performance":
        st.subheader("Predict Student Final Grade")
        st.markdown("Enter student information to predict their final grade.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Personal Information**")
            age = st.number_input("Age", min_value=18, max_value=25, value=20, step=1)
            gender = st.selectbox("Gender", ["Male", "Female"])
            attendance = st.slider("Attendance (%)", 0, 100, 85, 1)
            study_hours = st.slider("Study Hours per Week", 0, 40, 15, 1)
            previous_gpa = st.slider("Previous GPA", 0.0, 4.0, 3.0, 0.1)
        
        with col2:
            st.markdown("**Academic Scores**")
            assignment_1 = st.slider("Assignment 1", 0, 100, 75, 1)
            assignment_2 = st.slider("Assignment 2", 0, 100, 75, 1)
            assignment_3 = st.slider("Assignment 3", 0, 100, 75, 1)
            assignment_4 = st.slider("Assignment 4", 0, 100, 75, 1)
            test_1 = st.slider("Test 1", 0, 100, 70, 1)
            test_2 = st.slider("Test 2", 0, 100, 70, 1)
        
        col3, col4 = st.columns(2)
        with col3:
            midterm = st.slider("Midterm Score", 0, 100, 72, 1)
        with col4:
            final_exam = st.slider("Final Exam Score", 0, 100, 70, 1)
        
        st.markdown("---")
        if st.button("🔮 Predict Grade", type="primary", use_container_width=True):
            with st.spinner("Processing prediction..."):
                # Calculate averages
                avg_assignment = (assignment_1 + assignment_2 + assignment_3 + assignment_4) / 4
                avg_test = (test_1 + test_2 + midterm + final_exam) / 4
                
                # Create input data
                input_data = pd.DataFrame({
                    'Age': [age],
                    'Gender': [gender],
                    'Attendance': [attendance],
                    'Assignment_1': [assignment_1],
                    'Assignment_2': [assignment_2],
                    'Assignment_3': [assignment_3],
                    'Assignment_4': [assignment_4],
                    'Test_1': [test_1],
                    'Test_2': [test_2],
                    'Midterm': [midterm],
                    'Final_Exam': [final_exam],
                    'Study_Hours_Per_Week': [study_hours],
                    'Previous_GPA': [previous_gpa],
                    'Avg_Assignment': [avg_assignment],
                    'Avg_Test': [avg_test]
                })
                
                # Preprocess
                temp_df = pd.concat([df.copy(), input_data], ignore_index=True)
                X_input, _, _ = preprocess_data(temp_df, st.session_state.preprocessor)
                X_input = X_input[-1:].reshape(1, -1)
                
                # Predict
                predicted_grade = st.session_state.lr_model.predict(X_input)[0]
                
                # Determine letter grade
                if predicted_grade >= 90:
                    letter_grade = "A"
                elif predicted_grade >= 80:
                    letter_grade = "B"
                elif predicted_grade >= 70:
                    letter_grade = "C"
                elif predicted_grade >= 60:
                    letter_grade = "D"
                else:
                    letter_grade = "F"
                
                st.success("✅ Prediction completed successfully!")
                
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Grade", f"{predicted_grade:.2f}")
                with col2:
                    st.metric("Letter Grade", letter_grade)
                with col3:
                    # Predict cluster
                    cluster = st.session_state.kmeans_model.predict(X_input)[0]
                    st.metric("Performance Cluster", f"Cluster {cluster}")
                
                # Visualize prediction
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=predicted_grade,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Final Grade"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'steps': [
                            {'range': [0, 60], 'color': "lightgray"},
                            {'range': [60, 70], 'color': "yellow"},
                            {'range': [70, 80], 'color': "lightblue"},
                            {'range': [80, 90], 'color': "lightgreen"},
                            {'range': [90, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    # Analytics Page
    elif page == "Analytics":
        st.subheader("Data Analytics & Visualizations")
        
        # Correlation analysis
        st.markdown("#### Feature Correlation with Final Grade")
        numeric_df = df.select_dtypes(include=[np.number])
        correlations = numeric_df.corr()['Final_Grade'].sort_values(ascending=False)
        correlations = correlations.drop('Final_Grade')
        
        fig = px.bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            labels={'x': 'Correlation Coefficient', 'y': 'Feature'},
            color=correlations.values,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Feature vs Performance
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Attendance vs Final Grade")
            fig = px.scatter(
                df,
                x='Attendance',
                y='Final_Grade',
                labels={'Attendance': 'Attendance (%)', 'Final_Grade': 'Final Grade'}
            )
            # Add trend line manually
            valid_data = df[['Attendance', 'Final_Grade']].dropna()
            if len(valid_data) > 1:
                z = np.polyfit(valid_data['Attendance'], valid_data['Final_Grade'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(valid_data['Attendance'].min(), valid_data['Attendance'].max(), 100)
                fig.add_scatter(x=x_trend, y=p(x_trend), mode='lines', name='Trend', 
                              line=dict(color='red', dash='dash'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Study Hours vs Final Grade")
            fig = px.scatter(
                df,
                x='Study_Hours_Per_Week',
                y='Final_Grade',
                labels={'Study_Hours_Per_Week': 'Study Hours/Week', 'Final_Grade': 'Final Grade'}
            )
            # Add trend line manually
            valid_data = df[['Study_Hours_Per_Week', 'Final_Grade']].dropna()
            if len(valid_data) > 1:
                z = np.polyfit(valid_data['Study_Hours_Per_Week'], valid_data['Final_Grade'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(valid_data['Study_Hours_Per_Week'].min(), valid_data['Study_Hours_Per_Week'].max(), 100)
                fig.add_scatter(x=x_trend, y=p(x_trend), mode='lines', name='Trend', 
                              line=dict(color='red', dash='dash'))
            st.plotly_chart(fig, use_container_width=True)
        
        # Clustering visualization
        st.markdown("---")
        st.markdown("#### Student Performance Clusters")
        
        df_clustered = df.copy()
        df_clustered['Cluster'] = st.session_state.cluster_labels
        
        # PCA visualization
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(st.session_state.X)
        
        fig = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            color=df_clustered['Cluster'].astype(str),
            labels={
                'x': f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})',
                'y': f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})',
                'color': 'Cluster'
            },
            title='Student Clusters (PCA Visualization)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster comparison
        st.markdown("#### Cluster Characteristics")
        cluster_stats = df_clustered.groupby('Cluster').agg({
            'Final_Grade': ['mean', 'count'],
            'Attendance': 'mean',
            'Study_Hours_Per_Week': 'mean',
            'Avg_Assignment': 'mean',
            'Avg_Test': 'mean'
        }).round(2)
        
        st.dataframe(cluster_stats, use_container_width=True)
        
        # Cluster comparison chart
        fig = px.bar(
            df_clustered,
            x='Cluster',
            y='Final_Grade',
            color='Cluster',
            labels={'Cluster': 'Cluster', 'Final_Grade': 'Average Final Grade'},
            title='Average Final Grade by Cluster'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights Page
    elif page == "Insights":
        st.subheader("Key Insights & Recommendations")
        
        # Model performance
        y_pred = st.session_state.lr_model.predict(st.session_state.X)
        r2 = r2_score(st.session_state.y, y_pred)
        rmse = np.sqrt(mean_squared_error(st.session_state.y, y_pred))
        
        st.markdown("#### Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score", f"{r2:.4f}")
        with col2:
            st.metric("RMSE", f"{rmse:.4f}")
        
        st.markdown("---")
        
        # Key findings
        st.markdown("#### Key Findings")
        
        numeric_df = df.select_dtypes(include=[np.number])
        top_factors = numeric_df.corr()['Final_Grade'].abs().sort_values(ascending=False).drop('Final_Grade')
        
        st.write("**Top factors affecting student performance:**")
        for i, (factor, corr) in enumerate(top_factors.head(5).items(), 1):
            st.write(f"{i}. **{factor}**: {corr:.3f} correlation")
        
        st.markdown("---")
        
        # Cluster insights
        st.markdown("#### Student Cluster Analysis")
        
        df_clustered = df.copy()
        df_clustered['Cluster'] = st.session_state.cluster_labels
        
        n_clusters = len(np.unique(st.session_state.cluster_labels))
        for cluster in range(n_clusters):
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster]
            with st.expander(f"Cluster {cluster} ({len(cluster_data)} students)"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Average Grade:** {cluster_data['Final_Grade'].mean():.2f}")
                with col2:
                    st.write(f"**Average Attendance:** {cluster_data['Attendance'].mean():.1f}%")
                with col3:
                    st.write(f"**Average Study Hours:** {cluster_data['Study_Hours_Per_Week'].mean():.1f}")
                
                grade_dist = cluster_data['Grade_Letter'].value_counts()
                st.write("**Grade Distribution:**")
                for grade, count in grade_dist.items():
                    st.write(f"- Grade {grade}: {count} students")
        
        st.markdown("---")
        
        # Recommendations
        st.markdown("#### Recommendations")
        
        st.info("""
        **1. Attendance Monitoring**
        - Students with higher attendance show significantly better performance
        - Implement attendance tracking systems
        - Provide incentives for regular attendance
        
        **2. Study Hours Enhancement**
        - Study hours are strongly correlated with academic success
        - Encourage effective study habits
        - Provide time management workshops
        
        **3. Assignment Performance**
        - Consistent assignment completion is crucial
        - Break down large assignments into manageable tasks
        - Provide timely feedback
        
        **4. Early Intervention**
        - Use clustering to identify at-risk students
        - Provide personalized support based on cluster characteristics
        - Monitor students in lower-performing clusters
        
        **5. Predictive Analytics**
        - Use the prediction model to forecast student performance
        - Allocate resources based on predictive insights
        - Implement proactive intervention strategies
        """)

if __name__ == "__main__":
    main()

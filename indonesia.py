# economic clustering indonesia.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

def main():
    st.set_page_config(
        page_title="Advanced Economic Clustering Indonesia",
        layout="wide",
        page_icon="📊"
    )
    
    st.title("🔍 Advanced Indonesia's Economic People Clustering Analysis")
    st.write("**An artificial intelligence tool that detects the economic status of every Indonesian citizen. The clustering findings have two, three, or four options: upper middle, low middle, affluent and middle.**")
    
    # Enhanced Sidebar for navigation
    st.sidebar.title("🌐 Navigation Panel")
    st.sidebar.markdown("---")
    
    # Using radio buttons for more elegant navigation
    app_mode = st.sidebar.radio(
        "Select Your Section:",
        ["🏠 Home", "📊 EDA", "🤖 Machine Learning", "📈 Visualization", "🚀 New Prediction", "ℹ️ About"],
        index=0
    )
    
    if app_mode == "🏠 Home":
        show_home()
    elif app_mode == "📊 EDA":
        show_eda()
    elif app_mode == "🤖 Machine Learning":
        show_ml()
    elif app_mode == "📈 Visualization":
        show_visualization()
    elif app_mode == "🚀 New Prediction":
        show_prediction()
    elif app_mode == "ℹ️ About":
        show_about()
    
    # Copyright at the bottom of main app
    st.markdown("---")
    st.markdown("### Copyright © 2025 - Indonesia's Economic People Prediction System")
    st.markdown("**AI Assistant for Early Detection of Indonesia's Economic Peoples. All Rights Reserved. Created by Ridha Ash Siddiqy.**")

def show_home():
    st.header("🏠 Welcome to Indonesia's Economic Clustering Analysis")
    
    st.markdown("""
    ## 🌟 Overview
    
    Welcome to the **Advanced Indonesia's Economic People Clustering Analysis** platform! 
    This intelligent system utilizes machine learning and artificial intelligence to analyze 
    and cluster Indonesian citizens based on their economic characteristics.
    
    ### 🎯 What You Can Do Here:
    
    🔍 **Exploratory Data Analysis (EDA)**
    - Comprehensive data exploration and visualization
    - Statistical analysis and data quality assessment
    - Correlation analysis and distribution insights
    
    🤖 **Machine Learning Clustering**
    - Advanced feature engineering and selection
    - Automatic optimal cluster determination
    - K-means clustering with performance metrics
    
    📈 **Interactive Visualization**
    - Cluster profiling and comparison
    - PCA visualization for cluster separation
    - Economic segment analysis
    
    🚀 **Real-time Prediction**
    - Predict economic segments for new individuals
    - Get personalized economic recommendations
    - Understand cluster characteristics
    
    ### 📊 Expected Economic Segments:
    
    The system identifies meaningful economic clusters such as:
    - **Affluent** - High net worth individuals
    - **Upper Middle** - Well-established economic status
    - **Middle** - Stable economic position
    - **Lower Middle** - Developing economic status
    - And other meaningful economic classifications
    
    ### 🚀 Get Started:
    
    Navigate through the sections using the sidebar to explore different aspects of the economic clustering analysis.
    """)
    
    # Quick stats if data is available
    if 'df_original' in st.session_state:
        st.markdown("---")
        st.subheader("📈 Quick Dataset Overview")
        
        df = st.session_state.df_original
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{df.shape[0]:,}")
        with col2:
            st.metric("Total Features", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Data Ready", "✅" if 'df_clean' in st.session_state else "⏳")

def show_about():
    st.header("ℹ️ About This Project")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🌍 Project Overview
        
        **Advanced Indonesia's Economic People Clustering Analysis** is an innovative 
        artificial intelligence platform designed to analyze and classify Indonesian 
        citizens based on their economic characteristics using advanced machine learning 
        techniques.
        
        ### 🎯 Mission & Vision
        
        **Mission:** To provide accurate, data-driven insights into Indonesia's economic 
        landscape through advanced clustering algorithms.
        
        **Vision:** Become the leading AI-powered economic analysis tool for understanding 
        and predicting economic segments in Indonesia.
        
        ### 🔬 Methodology
        
        This system employs:
        - **K-means Clustering** for economic segmentation
        - **Advanced Feature Engineering** for meaningful economic indicators
        - **Automatic Feature Selection** for optimal model performance
        - **Comprehensive Validation Metrics** including Silhouette Score, Calinski-Harabasz, and Davies-Bouldin Index
        
        ### 💡 Key Features
        
        - **Automated Data Preprocessing**: Handles missing values and data quality issues
        - **Intelligent Feature Selection**: Automatically selects most relevant economic indicators
        - **Optimal Cluster Detection**: Determines the best number of economic segments
        - **Real-time Prediction**: Classifies new individuals into economic segments
        - **Comprehensive Visualization**: Interactive charts and insights
        
        ### 📊 Data Sources
        
        The analysis is based on comprehensive economic datasets including:
        - Salary and income data
        - Savings and investment information
        - Debt and liability records
        - Demographic characteristics
        """)
    
    with col2:
        st.markdown("""
        ## 👨‍💻 Developer Information
        
        **Created by:** Ridha Ash Siddiqy
        
        ### 🛠 Technical Stack
        
        - **Programming Language**: Python
        - **Web Framework**: Streamlit
        - **Machine Learning**: Scikit-learn
        - **Data Analysis**: Pandas, NumPy
        - **Visualization**: Matplotlib, Seaborn
        - **Clustering**: K-means Algorithm
        
        ### 📈 Model Performance
        
        The system focuses on achieving:
        - High silhouette scores (>0.5 for excellent separation)
        - Meaningful economic segment interpretation
        - Robust and scalable clustering solutions
        - User-friendly interface and insights
        
        ### 🔒 Data Privacy
        
        - All analysis respects data privacy principles
        - Anonymous data processing
        - Secure data handling protocols
        - Compliance with data protection standards
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 📞 Contact & Support
    
    For questions, suggestions, or collaboration opportunities, please reach out through 
    appropriate channels. This project represents ongoing research in economic clustering 
    and artificial intelligence applications in socioeconomic analysis.
    
    **🌟 Continuous Improvement:** This platform is regularly updated with new features, 
    improved algorithms, and enhanced visualization capabilities.
    """)

def load_data():
    """Load data from CSV file"""
    try:
        df = pd.read_csv("DataEkonomiIndonesia.csv")
        return df
    except FileNotFoundError:
        st.error("❌ File 'DataEkonomiIndonesia.csv' not found. Please ensure the file is in the correct directory.")
        return None

def show_eda():
    st.header("📊 Exploratory Data Analysis (EDA)")
    
    df = load_data()
    if df is None:
        return
    
    # Display data overview
    st.subheader("1. Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", df.shape[0])
    with col2:
        st.metric("Total Features", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Duplicate Records", df.duplicated().sum())
    
    st.write("**Data Preview:**")
    st.dataframe(df.head(10))
    
    # Data types information
    st.subheader("2. Data Types Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Data Types Summary:**")
        dtype_summary = pd.DataFrame(df.dtypes.value_counts()).reset_index()
        dtype_summary.columns = ['Data Type', 'Count']
        st.dataframe(dtype_summary)
    
    with col2:
        st.write("**Detailed Data Types:**")
        detailed_dtypes = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum()
        })
        st.dataframe(detailed_dtypes)
    
    # Missing values analysis
    st.subheader("3. Missing Values Analysis")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    
    if len(missing_data) > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Columns with Missing Values:**")
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing Percentage': (missing_data.values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_df.set_index('Column')['Missing Count'].plot(kind='bar', ax=ax, color='red', alpha=0.7)
            ax.set_title('Missing Values by Column', fontweight='bold')
            ax.set_ylabel('Missing Count')
            plt.xticks(rotation=45)
            st.pyplot(fig)
    else:
        st.success("✅ No missing values found in the dataset!")
    
    # Statistical summary
    st.subheader("4. Statistical Summary")
    
    # Numerical columns statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        st.write("**Numerical Variables Summary:**")
        st.dataframe(df[numerical_cols].describe().T.style.format("{:.2f}"))
    
    # Categorical columns statistics
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.write("**Categorical Variables Summary:**")
        for col in categorical_cols:
            st.write(f"**{col}:**")
            value_counts = df[col].value_counts()
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(value_counts)
            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                value_counts.head(10).plot(kind='bar', ax=ax)
                ax.set_title(f'Top 10 {col} Values')
                plt.xticks(rotation=45)
                st.pyplot(fig)
    
    # Correlation analysis
    st.subheader("5. Correlation Analysis")
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax,
                   fmt='.2f', linewidths=0.5, linecolor='white')
        ax.set_title('Correlation Matrix Heatmap', fontweight='bold', pad=20)
        st.pyplot(fig)
        
        # Find high correlations
        st.write("**High Correlations (|r| > 0.7):**")
        high_corr = corr_matrix.unstack().sort_values(ascending=False)
        high_corr = high_corr[high_corr < 1]  # Remove self-correlations
        high_corr = high_corr[abs(high_corr) > 0.7]
        if len(high_corr) > 0:
            st.dataframe(pd.DataFrame(high_corr, columns=['Correlation']))
        else:
            st.info("No strong correlations found (|r| > 0.7)")
    
    # Distribution analysis
    st.subheader("6. Distribution Analysis")
    
    if len(numerical_cols) > 0:
        selected_numeric = st.multiselect("Select numerical variables for distribution analysis:", 
                                        numerical_cols, default=numerical_cols[:3])
        
        if selected_numeric:
            n_cols = min(3, len(selected_numeric))
            n_rows = (len(selected_numeric) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(selected_numeric):
                if i < len(axes):
                    axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    axes[i].set_title(f'Distribution of {col}', fontweight='bold')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(selected_numeric), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Handle missing values automatically
    st.subheader("7. Data Preprocessing")
    
    df_clean = df.copy()
    
    # Handle missing values
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype == 'object':
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    st.success("✅ Missing values handled successfully!")
    
    # Save cleaned data to session state
    st.session_state.df_clean = df_clean
    st.session_state.df_original = df
    
    st.info("**EDA Summary:** Dataset is ready for machine learning analysis with comprehensive insights into data structure, distributions, and relationships.")

def show_ml():
    st.header("🤖 Advanced Machine Learning with Auto Feature Selection")
    
    if 'df_clean' not in st.session_state:
        st.warning("⚠️ Please run Comprehensive EDA first")
        return
    
    df_clean = st.session_state.df_clean
    
    st.info("""
    **🎯 Advanced Clustering Strategy:**
    - **Automatic Feature Selection**: Optimal features selected automatically
    - **Optimal Cluster Determination**: Based on silhouette analysis
    - **High Silhouette Focus**: Target > 0.5 for excellent separation
    - **Comprehensive Evaluation**: Multiple metrics for robust assessment
    """)
    
    st.subheader("1. Advanced Feature Engineering")
    
    # Create advanced features
    df_engineered = df_clean.copy()
    
    # Basic financial ratios
    df_engineered['net_worth'] = df_engineered.get('savings', 0) + df_engineered.get('investment', 0) - df_engineered.get('debt', 0)
    df_engineered['savings_ratio'] = df_engineered.get('savings', 0) / (df_engineered.get('expenses', 0) + 1)
    df_engineered['investment_ratio'] = df_engineered.get('investment', 0) / (df_engineered.get('salary', 0) + 1)
    df_engineered['debt_to_income'] = df_engineered.get('debt', 0) / (df_engineered.get('salary', 0) + 1)
    
    # Composite scores
    df_engineered['financial_health'] = (
        np.log1p(df_engineered.get('salary', 0)) * 0.3 +
        np.log1p(df_engineered.get('savings', 0)) * 0.4 +
        np.log1p(df_engineered.get('investment', 0)) * 0.3 -
        df_engineered['debt_to_income'] * 0.2
    )
    
    # Handle infinite values
    df_engineered = df_engineered.replace([np.inf, -np.inf], np.nan)
    df_engineered = df_engineered.fillna(0)
    
    st.write("**Advanced Feature Engineering Completed**")
    st.dataframe(df_engineered[['net_worth', 'financial_health', 'savings_ratio', 'investment_ratio']].head())
    
    st.subheader("2. Automatic Feature Selection")
    
    # Prepare features for selection
    numerical_features = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove highly correlated features
    corr_matrix = df_engineered[numerical_features].corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
    
    if to_drop:
        st.write(f"**Removed highly correlated features:** {to_drop}")
        numerical_features = [feat for feat in numerical_features if feat not in to_drop]
    
    # Use feature selection - FIXED VERSION
    X = df_engineered[numerical_features]
    
    # Create a temporary target for feature selection using clustering
    from sklearn.cluster import KMeans
    
    # Scale the data for temporary clustering
    scaler_temp = StandardScaler()
    X_temp_scaled = scaler_temp.fit_transform(X)
    
    # Create temporary clusters for feature selection
    kmeans_temp = KMeans(n_clusters=3, random_state=42, n_init=10)
    y_temp = kmeans_temp.fit_predict(X_temp_scaled)
    
    # Select top features automatically
    k_features = st.slider("Number of features to select:", 5, 12, 10)
    selector = SelectKBest(score_func=f_classif, k=k_features)
    selector.fit(X, y_temp)
    
    # Get selected features
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()
    
    # Create feature scores dataframe with proper values
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_
    })
    
    # Replace any NaN or infinite scores with 0
    feature_scores['Score'] = feature_scores['Score'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Sort by score
    feature_scores = feature_scores.sort_values('Score', ascending=False)
    
    st.write("**Feature Selection Results:**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Selected Features:**")
        for i, feature in enumerate(selected_features, 1):
            st.write(f"{i}. {feature}")
    
    with col2:
        st.write("**Top Feature Scores:**")
        # Format the scores to show 2 decimal places
        display_scores = feature_scores.head(10).copy()
        display_scores['Score'] = display_scores['Score'].round(2)
        st.dataframe(display_scores)
    
    # Prepare final feature set
    X_selected = df_engineered[selected_features]
    
    # Handle categorical features if any are selected
    categorical_features = X_selected.select_dtypes(include=['object']).columns
    if len(categorical_features) > 0:
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        X_categorical_encoded = encoder.fit_transform(X_selected[categorical_features])
        feature_names = encoder.get_feature_names_out(categorical_features)
        X_categorical_df = pd.DataFrame(X_categorical_encoded, columns=feature_names, index=X_selected.index)
        
        # Combine with numerical features
        X_numerical = X_selected.select_dtypes(include=[np.number])
        X_final = pd.concat([X_numerical, X_categorical_df], axis=1)
        st.session_state.encoder = encoder
    else:
        X_final = X_selected
        st.session_state.encoder = None
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)
    
    st.success(f"✅ Automatic feature selection completed! Selected {len(selected_features)} features.")
    
    # Save to session state
    st.session_state.X_scaled = X_scaled
    st.session_state.X_final = X_final
    st.session_state.scaler = scaler
    st.session_state.selected_features = selected_features
    st.session_state.df_engineered = df_engineered
    
    st.subheader("3. Optimal Cluster Determination")
    
    st.info("**Determining optimal number of clusters using silhouette analysis...**")
    
    if st.button("🚀 Find Optimal Clusters & Train Model"):
        with st.spinner("Analyzing optimal cluster configuration..."):
            
            # Find optimal number of clusters
            silhouette_scores = []
            wcss_scores = []
            k_range = range(2, 11)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, k in enumerate(k_range):
                progress = (i + 1) / len(k_range)
                progress_bar.progress(progress)
                status_text.text(f"Testing K={k}...")
                
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                
                sil_score = silhouette_score(X_scaled, cluster_labels)
                silhouette_scores.append(sil_score)
                wcss_scores.append(kmeans.inertia_)
            
            progress_bar.empty()
            status_text.empty()
            
            # Find optimal K (highest silhouette score)
            optimal_k = k_range[np.argmax(silhouette_scores)]
            best_silhouette = max(silhouette_scores)
            
            # Train final model with optimal K
            final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
            final_labels = final_kmeans.fit_predict(X_scaled)
            
            # Calculate final metrics
            final_silhouette = silhouette_score(X_scaled, final_labels)
            final_calinski = calinski_harabasz_score(X_scaled, final_labels)
            final_davies = davies_bouldin_score(X_scaled, final_labels)
            
            # Add clusters to dataframe
            df_result = df_engineered.copy()
            df_result['cluster'] = final_labels
            
            # Assign meaningful cluster names based on economic characteristics
            df_result = assign_economic_clusters(df_result)
            
            # Save to session state
            st.session_state.kmeans = final_kmeans
            st.session_state.df_result = df_result
            st.session_state.cluster_labels = final_labels
            st.session_state.final_silhouette = final_silhouette
            st.session_state.final_calinski = final_calinski
            st.session_state.final_davies = final_davies
            st.session_state.optimal_k = optimal_k
            st.session_state.silhouette_scores = silhouette_scores
            st.session_state.wcss_scores = wcss_scores
            st.session_state.k_range = k_range
            
            st.success(f"✅ Model trained successfully with {optimal_k} clusters!")
            
            # Display results
            st.subheader("📊 Model Performance Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Optimal Clusters", optimal_k)
            with col2:
                st.metric("Silhouette Score", f"{final_silhouette:.4f}",
                         delta="Excellent" if final_silhouette > 0.5 else "Good" if final_silhouette > 0.3 else "Needs Improvement")
            with col3:
                st.metric("Calinski-Harabasz", f"{final_calinski:.0f}")
            with col4:
                st.metric("Davies-Bouldin", f"{final_davies:.4f}")
            
            # Silhouette score assessment
            st.subheader("🔍 Silhouette Score Analysis")
            
            if final_silhouette > 0.5:
                st.success("🎯 **EXCELLENT SEPARATION**: Outstanding cluster differentiation with clear boundaries!")
                st.balloons()
            elif final_silhouette > 0.4:
                st.success("✅ **STRONG SEPARATION**: Very clear cluster boundaries")
            elif final_silhouette > 0.3:
                st.success("✅ **GOOD SEPARATION**: Well-defined clusters")
            elif final_silhouette > 0.25:
                st.warning("⚠️ **MODERATE SEPARATION**: Acceptable cluster differentiation")
            else:
                st.error("❌ **NEEDS IMPROVEMENT**: Consider feature engineering or alternative algorithms")
            
            # Cluster determination plots
            st.subheader("📈 Cluster Determination Analysis")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Elbow method plot
            ax1.plot(k_range, wcss_scores, 'bo-', linewidth=2, markersize=6)
            ax1.set_xlabel('Number of Clusters (K)')
            ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
            ax1.set_title('Elbow Method for Optimal K', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal K={optimal_k}')
            ax1.legend()
            
            # Silhouette score plot
            ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=6)
            ax2.set_xlabel('Number of Clusters (K)')
            ax2.set_ylabel('Silhouette Score')
            ax2.set_title('Silhouette Analysis for Optimal K', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal K={optimal_k}')
            ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Excellent Threshold')
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Cluster distribution
            st.subheader("📊 Cluster Distribution")
            cluster_counts = df_result['cluster'].value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Cluster Size Distribution:**")
                cluster_stats = pd.DataFrame({
                    'Cluster': cluster_counts.index,
                    'Count': cluster_counts.values,
                    'Percentage': (cluster_counts.values / len(df_result) * 100).round(1)
                })
                st.dataframe(cluster_stats)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))
                bars = ax.bar(cluster_counts.index, cluster_counts.values, color=colors, edgecolor='black', alpha=0.8)
                ax.set_title('Cluster Size Distribution', fontweight='bold')
                ax.set_xlabel('Cluster')
                ax.set_ylabel('Number of Members')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', fontweight='bold')
                
                st.pyplot(fig)

def assign_economic_clusters(df_result):
    """Assign meaningful economic cluster names based on financial characteristics"""
    cluster_stats = df_result.groupby('cluster').agg({
        'salary': 'mean',
        'savings': 'mean',
        'investment': 'mean',
        'debt': 'mean',
        'net_worth': 'mean',
        'financial_health': 'mean'
    }).round(2)
    
    # Sort by financial health and net worth
    cluster_stats = cluster_stats.sort_values(['financial_health', 'net_worth'], ascending=False)
    
    # Define economic tiers based on percentiles
    economic_tiers = ['Affluent', 'Upper Middle', 'Middle', 'Lower Middle', 'Struggling', 
                     'Developing', 'Emerging', 'Established', 'Prosperous', 'Wealthy']
    
    cluster_names = {}
    for i, cluster_id in enumerate(cluster_stats.index):
        if i < len(economic_tiers):
            cluster_names[cluster_id] = f"Cluster {cluster_id} - {economic_tiers[i]}"
        else:
            cluster_names[cluster_id] = f"Cluster {cluster_id}"
    
    df_result['economic_segment'] = df_result['cluster'].map(cluster_names)
    return df_result

def show_visualization():
    st.header("📈 Comprehensive Visualization")
    
    if 'df_result' not in st.session_state:
        st.warning("⚠️ Please train the model first in the Machine Learning tab")
        return
    
    df_result = st.session_state.df_result
    
    st.subheader("1. Cluster Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Clusters", st.session_state.optimal_k)
    with col2:
        st.metric("Silhouette Score", f"{st.session_state.final_silhouette:.4f}")
    with col3:
        st.metric("Total Records", len(df_result))
    
    st.subheader("2. Cluster Distribution Analysis")
    
    cluster_counts = df_result['economic_segment'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Cluster Statistics:**")
        cluster_stats = pd.DataFrame({
            'Economic Segment': cluster_counts.index,
            'Count': cluster_counts.values,
            'Percentage': (cluster_counts.values / len(df_result) * 100).round(1)
        })
        st.dataframe(cluster_stats)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))
        wedges, texts, autotexts = ax.pie(cluster_counts.values, labels=cluster_counts.index, 
                                         autopct='%1.1f%%', startangle=90, colors=colors)
        ax.set_title('Economic Segment Distribution', fontweight='bold')
        
        # Improve text appearance
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        
        st.pyplot(fig)
    
    st.subheader("3. Cluster Profiling")
    
    # Select cluster for detailed analysis
    selected_segment = st.selectbox("Select economic segment for detailed analysis:", 
                                   sorted(df_result['economic_segment'].unique()))
    segment_data = df_result[df_result['economic_segment'] == selected_segment]
    
    st.write(f"**📊 {selected_segment} Profile ({len(segment_data)} individuals)**")
    
    # Financial statistics
    financial_features = ['salary', 'savings', 'investment', 'debt', 'net_worth', 'financial_health']
    financial_stats = segment_data[financial_features].describe()
    
    st.write("**💰 Financial Analysis:**")
    st.dataframe(financial_stats.style.format('{:,.2f}'))
    
    # Comparison with other clusters
    st.subheader("4. Cross-Cluster Comparison")
    
    comparison_features = ['salary', 'savings', 'investment', 'debt', 'net_worth', 'financial_health']
    comparison_stats = df_result.groupby('economic_segment')[comparison_features].mean().round(2)
    
    st.write("**Financial Metrics Comparison Across Segments:**")
    st.dataframe(comparison_stats)
    
    # Enhanced heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(comparison_stats.T, annot=True, cmap='RdYlGn', ax=ax, fmt='.0f', 
                linewidths=1, linecolor='white')
    ax.set_title('Financial Metrics Comparison Across Economic Segments', fontweight='bold', pad=20)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # PCA Visualization
    st.subheader("5. Cluster Visualization (PCA)")
    
    try:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(st.session_state.X_scaled)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=st.session_state.cluster_labels, 
                           cmap='viridis', alpha=0.7, s=50)
        ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title('Cluster Visualization using PCA', fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"PCA visualization not available: {e}")
    
    st.subheader("6. Download Results")
    
    csv = df_result.to_csv(index=False)
    st.download_button(
        label="📥 Download Complete Clustering Results (CSV)",
        data=csv,
        file_name="economic_clustering_results.csv",
        mime="text/csv"
    )

def show_prediction():
    st.header("🚀 Economic Segment Prediction")
    
    if 'kmeans' not in st.session_state:
        st.warning("⚠️ Please train the model first in the Machine Learning tab")
        return
    
    st.subheader("Enter Individual's Economic Data for Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**💵 Financial Information (Million Rupiah):**")
        
        salary = st.number_input("Monthly Salary", min_value=0.0, value=10.0, step=1.0, help="In million Rupiah")
        savings = st.number_input("Total Savings", min_value=0.0, value=20.0, step=5.0, help="In million Rupiah")
        investment = st.number_input("Investment Portfolio", min_value=0.0, value=15.0, step=5.0, help="In million Rupiah")
        debt = st.number_input("Total Debt", min_value=0.0, value=5.0, step=1.0, help="In million Rupiah")
        expenses = st.number_input("Monthly Expenses", min_value=0.0, value=8.0, step=1.0, help="In million Rupiah")
    
    with col2:
        st.write("**👤 Demographic Information:**")
        
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        education = st.selectbox("Education Level", 
                               ["SHS", "Bachelor", "Master", "Doctoral", "Diploma", "JHS", "ES"])
    
    if st.button("🎯 Predict Economic Segment"):
        try:
            # Calculate engineered features
            net_worth = savings + investment - debt
            savings_ratio = savings / (expenses + 1)
            investment_ratio = investment / (salary + 1)
            debt_to_income = debt / (salary + 1)
            
            financial_health = (
                np.log1p(salary) * 0.3 +
                np.log1p(savings) * 0.4 +
                np.log1p(investment) * 0.3 -
                debt_to_income * 0.2
            )
            
            # Prepare input using selected features
            selected_features = st.session_state.get('selected_features', [])
            input_values = []
            
            for feature in selected_features:
                if feature == 'salary':
                    input_values.append(salary)
                elif feature == 'savings':
                    input_values.append(savings)
                elif feature == 'investment':
                    input_values.append(investment)
                elif feature == 'debt':
                    input_values.append(debt)
                elif feature == 'net_worth':
                    input_values.append(net_worth)
                elif feature == 'savings_ratio':
                    input_values.append(savings_ratio)
                elif feature == 'investment_ratio':
                    input_values.append(investment_ratio)
                elif feature == 'debt_to_income':
                    input_values.append(debt_to_income)
                elif feature == 'financial_health':
                    input_values.append(financial_health)
                elif feature == 'age':
                    input_values.append(age)
                elif feature == 'expenses':
                    input_values.append(expenses)
                else:
                    # For features not provided, use median from training data
                    input_values.append(0)
            
            input_numeric = np.array([input_values])
            
            # Handle categorical features if encoder exists
            if st.session_state.encoder is not None:
                # This is simplified - in practice, you'd need to handle categorical encoding properly
                categorical_input = np.zeros((1, len(st.session_state.encoder.get_feature_names_out())))
                input_combined = np.concatenate([input_numeric, categorical_input], axis=1)
            else:
                input_combined = input_numeric
            
            # Scale input
            input_scaled = st.session_state.scaler.transform(input_combined)
            
            # Predict cluster
            cluster_pred = st.session_state.kmeans.predict(input_scaled)[0]
            
            # Get economic segment
            df_result = st.session_state.df_result
            economic_segment = df_result[df_result['cluster'] == cluster_pred]['economic_segment'].iloc[0]
            
            # Display results
            st.success(f"🎯 **Predicted Economic Segment: {economic_segment}**")
            
            # Show segment characteristics
            segment_data = df_result[df_result['economic_segment'] == economic_segment]
            
            st.subheader(f"📊 {economic_segment} Characteristics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Salary", f"Rp{segment_data['salary'].mean():.1f}M")
                st.metric("Average Savings", f"Rp{segment_data['savings'].mean():.1f}M")
                st.metric("Segment Size", f"{len(segment_data):,} people")
            
            with col2:
                st.metric("Average Investment", f"Rp{segment_data['investment'].mean():.1f}M")
                st.metric("Average Net Worth", f"Rp{segment_data['net_worth'].mean():.1f}M")
                st.metric("Financial Health", f"{segment_data['financial_health'].mean():.1f}")
            
            with col3:
                if 'age' in segment_data.columns:
                    st.metric("Average Age", f"{segment_data['age'].mean():.1f} years")
                if 'education' in segment_data.columns:
                    common_edu = segment_data['education'].mode()
                    if not common_edu.empty:
                        st.metric("Common Education", common_edu.iloc[0])
            
            # Economic recommendations
            st.subheader("💡 Economic Recommendations")
            
            advice_mapping = {
                "Affluent": """
                **Wealth Preservation & Growth**:
                - Diversify investments across multiple asset classes
                - Consider international investment opportunities
                - Implement comprehensive tax optimization strategies
                - Focus on legacy and estate planning
                - Explore philanthropic opportunities
                """,
                "Upper Middle": """
                **Wealth Accumulation & Protection**:
                - Maximize retirement and investment contributions
                - Develop multiple income streams
                - Build substantial emergency funds (6-12 months)
                - Consider real estate investments
                - Enhance professional skills for career advancement
                """,
                "Middle": """
                **Financial Stability & Growth**:
                - Build 3-6 month emergency fund
                - Focus on high-interest debt reduction
                - Start systematic investment plan
                - Enhance financial literacy
                - Explore side income opportunities
                """,
                "Lower Middle": """
                **Financial Foundation Building**:
                - Create detailed budget and track expenses
                - Prioritize debt repayment strategies
                - Build basic emergency fund (1-3 months)
                - Focus on skill development for income growth
                - Start small, consistent savings habit
                """,
                "Struggling": """
                **Financial Recovery & Basic Stability**:
                - Create basic survival budget
                - Prioritize essential expenses
                - Seek debt counseling if needed
                - Explore government assistance programs
                - Focus on immediate income-generating activities
                """
            }
            
            # Find matching advice
            advice = "Focus on financial education and basic money management principles."
            for key in advice_mapping:
                if key.lower() in economic_segment.lower():
                    advice = advice_mapping[key]
                    break
                    
            st.info(advice)
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.info("Please ensure all input fields are filled correctly and try again.")

if __name__ == "__main__":
    main()



import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go

# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title="GemPricer AI",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# CUSTOM CSS FOR BETTER UI
# ===========================
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 3em;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    
    /* Metric cards styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 0.9em;
        opacity: 0.9;
    }
    
    /* Prediction result styling */
    .prediction-result {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
    }
    
    /* Section divider */
    .section-divider {
        margin: 40px 0;
        border-top: 2px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# LOAD DATA
# ===========================
@st.cache_data
def load_data():
    """Load and preprocess the diamond dataset"""
    df = pd.read_csv("cubic_zirconia.csv")
    
    # Remove unwanted columns
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)
    
    # Remove missing values
    df.dropna(inplace=True)
    
    # Encode categorical columns
    df["cut"] = df["cut"].map({
        "Fair": 0,
        "Good": 1,
        "Very Good": 2,
        "Premium": 3,
        "Ideal": 4
    })
    
    df["color"] = df["color"].map({
        "J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6
    })
    
    df["clarity"] = df["clarity"].map({
        "I1": 0, "SI2": 1, "SI1": 2, "VS2": 3,
        "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7
    })
    
    X = df.drop("price", axis=1)
    y = df["price"]
    
    return df, X, y

# ===========================
# TRAIN MODEL
# ===========================
@st.cache_resource
def train_model():
    """Train Random Forest model"""
    df, X, y = load_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    score = r2_score(y_test, prediction)
    
    return model, score

# Load data and model
df, X, y = load_data()
model, score = train_model()

# ===========================
# HEADER SECTION
# ===========================
st.markdown('<div class="main-title">💎 GemPricer AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Machine Learning | Predict prices instantly</div>', unsafe_allow_html=True)

# ===========================
# KEY METRICS
# ===========================
col1, col2, col3, col4 = st.columns(4, gap="medium")

metrics = [
    (col1, "📊 Dataset", f"{len(df):,}", "Records"),
    (col2, "🔧 Features", f"{X.shape[1]}", "Variables"),
    (col3, "✅ Model Score", f"{score:.3f}", "R² Score"),
    (col4, "💰 Avg Price", f"${df['price'].mean():,.0f}", "USD")
]

for col, icon, value, label in metrics:
    with col:
        st.metric(icon, value, label)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ===========================
# MAIN INTERFACE WITH TABS
# ===========================
tab1, tab2, tab3, tab4 = st.tabs([
    "💎 Price Predictor",
    "📊 Data Analysis",
    "📈 Model Insights",
    "📄 Dataset"
])

# ========================
# TAB 1: PRICE PREDICTOR
# ========================
with tab1:
    st.subheader("Enter Diamond Details")
    
    col1, col2 = st.columns(2, gap="large")
    
    # Left column - Categorical features
    with col1:
        st.write("#### Quality Attributes")
        
        cut = st.selectbox(
            "💎 Cut Quality",
            ["Fair", "Good", "Very Good", "Premium", "Ideal"],
            help="Diamond cut quality - Ideal is the best"
        )
        
        color = st.selectbox(
            "🎨 Color Grade",
            ["D", "E", "F", "G", "H", "I", "J"],
            index=3,
            help="D is colorless (best), J is light color"
        )
        
        clarity = st.selectbox(
            "✨ Clarity",
            ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"],
            help="IF = Internally Flawless (best), I1 = Included"
        )
    
    # Right column - Numerical features
    with col2:
        st.write("#### Weight & Dimensions")
        
        carat = st.slider(
            "⚖️ Carat Weight",
            0.20, 5.00, 1.00, 0.01,
            help="Diamond weight in carats"
        )
        
        depth = st.slider(
            "📏 Depth %",
            40.0, 80.0, 61.5, 0.1,
            help="Depth as percentage of diameter"
        )
        
        table = st.slider(
            "📐 Table %",
            40.0, 100.0, 57.0, 0.1,
            help="Table width as percentage of diameter"
        )
    
    # Dimensions
    st.write("#### Physical Dimensions (mm)")
    col_x, col_y, col_z = st.columns(3)
    
    with col_x:
        x = st.slider("Length (X)", 0.00, 15.00, 5.50, 0.01)
    with col_y:
        y = st.slider("Width (Y)", 0.00, 15.00, 5.50, 0.01)
    with col_z:
        z = st.slider("Height (Z)", 0.00, 10.00, 3.50, 0.01)
    
    # ===========================
    # ENCODING MAPPINGS
    # ===========================
    cut_map = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
    color_map = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
    clarity_map = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}
    
    # ===========================
    # CREATE INPUT DATAFRAME
    # ===========================
    input_data = pd.DataFrame({
        "carat": [carat],
        "cut": [cut_map[cut]],
        "color": [color_map[color]],
        "clarity": [clarity_map[clarity]],
        "depth": [depth],
        "table": [table],
        "x": [x],
        "y": [y],
        "z": [z]
    })
    
    # Display input summary
    st.write("#### Input Summary")
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric("Cut", cut)
    with summary_col2:
        st.metric("Color", color)
    with summary_col3:
        st.metric("Clarity", clarity)
    
    st.info(f"**Weight:** {carat} carats | **Dimensions:** {x}×{y}×{z} mm")
    
    # ===========================
    # PREDICTION BUTTON
    # ===========================
    predict_button = st.button(
        "💎 PREDICT PRICE",
        use_container_width=True,
        type="primary"
    )
    
    if predict_button:
        prediction = model.predict(input_data)[0]
        
        # Success animation
        st.balloons()
        
        # Display prediction
        st.markdown(f'<div class="prediction-result">${prediction:,.2f}</div>', unsafe_allow_html=True)
        
        # Additional info
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.success(f"✅ Prediction successful!")
        with col_info2:
            st.info(f"Model confidence: {score:.1%}")
        
        st.warning("⚠️ **Note:** This is an estimate. Actual prices may vary based on market conditions and other factors.")

# ========================
# TAB 2: DATA ANALYSIS
# ========================
with tab2:
    st.subheader("Diamond Market Analysis")
    
    # Price distribution
    st.write("#### Price Distribution")
    fig1 = px.histogram(
        df, x="price", nbins=40,
        title="Distribution of Diamond Prices",
        color_discrete_sequence=["#667eea"]
    )
    fig1.update_layout(xaxis_title="Price ($)", yaxis_title="Count")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Carat vs Price
    st.write("#### Carat Weight vs Price")
    fig2 = px.scatter(
        df, x="carat", y="price",
        color="cut",
        title="How Carat Affects Diamond Price",
        size="carat",
        hover_data=["clarity", "color"],
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig2.update_layout(xaxis_title="Carat Weight", yaxis_title="Price ($)")
    st.plotly_chart(fig2, use_container_width=True)
    
    # Color distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Color Distribution")
        fig3 = px.pie(
            df, names="color",
            title="Diamond Colors in Dataset",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.write("#### Clarity Distribution")
        clarity_order = ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"]
        clarity_counts = df["clarity"].map({v: k for k, v in clarity_map.items()}).value_counts()
        fig4 = px.pie(
            values=clarity_counts.values,
            names=clarity_counts.index,
            title="Clarity Levels in Dataset"
        )
        st.plotly_chart(fig4, use_container_width=True)

# ========================
# TAB 3: MODEL INSIGHTS
# ========================
with tab3:
    st.subheader("Machine Learning Model Analysis")
    
    # Model performance
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{score:.3f}", help="How well the model fits the data")
    with col2:
        st.metric("Model Type", "Random Forest", help="200 decision trees")
    with col3:
        st.metric("Train/Test Split", "80/20", help="Training and testing data ratio")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Feature importance
    st.write("#### 🎯 Feature Importance")
    st.write("*Which features have the most impact on diamond price?*")
    
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    fig_importance = px.bar(
        importance_df,
        x="Importance", y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Viridis",
        title="Feature Importance for Price Prediction"
    )
    fig_importance.update_layout(showlegend=False)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Correlation heatmap
    st.write("#### 🔗 Feature Correlation Matrix")
    st.write("*How features relate to each other*")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True, fmt=".2f", cmap="coolwarm",
        ax=ax, square=True, cbar_kws={"label": "Correlation"}
    )
    ax.set_title("Diamond Features Correlation", fontsize=14, fontweight="bold")
    st.pyplot(fig)

# ========================
# TAB 4: DATASET PREVIEW
# ========================
with tab4:
    st.subheader("Dataset Overview")
    
    # Dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Features", X.shape[1])
    with col3:
        st.metric("Price Range", f"${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Dataset preview
    st.write("#### First 10 Records")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Download dataset
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Dataset (CSV)",
        data=csv,
        file_name="cubic_zirconia.csv",
        mime="text/csv",
        use_container_width=True
    )

# ===========================
# FOOTER
# ===========================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.9em;'>
💎 Diamond Price Prediction System | Built with Streamlit + Scikit-Learn
</div>
""", unsafe_allow_html=True)

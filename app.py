import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, silhouette_score
import base64
import time

# Configure page
st.set_page_config(
    page_title="Krusty Krab Financial ML 🍔",
    page_icon="🐙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Welcome Page Configuration ---
WELCOME_CONFIG = {
    "background": "https://media.giphy.com/media/3o6wO7zlFQqwTebnck/giphy.gif?cid=ecf05e475pxwt8myzth0bwtzh7xxjc0v7pi8esi8pdac4ba9&ep=v1_gifs_search&rid=giphy.gif&ct=g",
    "characters": {
        "spongebob": "https://media.giphy.com/media/3o7abKhOpu0NwenH3O/giphy.gif",
        "patrick": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExc3ByNDdpNGdlbmpiaTFxenVueDRiOXR1ZXh4Y3I3bmNuM3FocHhobSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/Wvh1de6cFXcWc/giphy.gif",
        "squidward": "https://media.giphy.com/media/mvW2WgHHi6vOo/giphy.gif?cid=ecf05e47rgne813fqizn8cn1uovgnegt4f0bfujt2i1zs2yl&ep=v1_gifs_search&rid=giphy.gif&ct=g",
        "mrkrabs": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExY293cnZhcnSidjRlM3dvZmh2NnJkY2p5czljd3Zqazh6eXJ3eTRvaiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/C5cDXxEgBGNjy/giphy.gif",
        "gary": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExa2pxM3VwbHFoN3owNGpnazJkZmdmaGRlOWN0Y2dvNXVvZXJoOHVldiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/12SwAWiqchwlYk/giphy.gif"
    },
    "quotes": [
        ("SpongeBob", "I'm ready! I'm ready! I'm ready to predict stock prices!", "🧽"),
        ("Mr. Krabs", "Money money money! Show me the profits!", "🦀"),
        ("Squidward", "This better be worth my time...", "🐙"),
        ("Patrick", "Is this the Krusty Krab? No, this is machine learning!", "⭐"),
        ("Gary", "Meow... (Translation: Let's make smart investments)", "🐌")
    ]
}

# --- Main Theme Configuration ---
THEMES = {
    "🧟 Zombie Krusty Krab": {
        "emojis": ["🧽", "🍔", "🐙", "👻", "🦑"],
        "background": "https://media.giphy.com/media/l1EtlYgIFF2qamtAA/giphy.gif",
        "primary_color": "#00FF00",  # Toxic green
        "secondary_color": "#8B0000",  # Blood red
        "font_family": "'Creepster', cursive",  # Spooky font
        "model": "Logistic Regression",
        "quotes": [
            "ARRR! Ready to serve Krabby Patties to the undead! 🧟🍔",
            "This ain't your average Krusty Krab, this is the Krusty KRABBY!",
            "Don't drop the patty... THE ZOMBIES ARE COMING! 🧟‍♂️"
        ],
        "custom_css": """
            @import url('https://fonts.googleapis.com/css2?family=Creepster&display=swap');
            .stApp {
                background-blend-mode: multiply;
                background-color: rgba(20, 20, 20, 0.9) !important;
            }
            .main-container {
                border: 2px dashed #8B0000 !important;
                background: rgba(0, 0, 0, 0.8) !important;
            }
            .stButton>button {
                text-shadow: 0 0 5px #00FF00 !important;
                box-shadow: 0 0 10px #00FF00 !important;
            }
            .success-message {
                font-family: 'Creepster', cursive !important;
                text-shadow: 0 0 5px #8B0000 !important;
            }
        """
    },
    "⚡ Plankton's Quantum Lab": {
        "emojis": ["⚡", "🔬", "🧪", "🤖", "💻"],
        "background": "https://media.giphy.com/media/l1KuiaUIyx9KaN3bO/giphy.gif",
        "primary_color": "#00FFFF",  # Cyan
        "secondary_color": "#FF00FF",  # Magenta
        "font_family": "'Orbitron', sans-serif",  # Tech font
        "model": "Linear Regression",
        "quotes": [
            "Behold the power of QUANTUM KRABBY PATTIES! ⚛️",
            "Chum Bucket technology will RULE THE WORLD! 🌎",
            "My algorithms are 1% patty, 99% PLANKTON POWER! 💪"
        ],
        "custom_css": """
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
            .stApp {
                animation: bgPulse 10s infinite alternate;
            }
            @keyframes bgPulse {
                0% { background-color: rgba(0,0,20,0.8); }
                100% { background-color: rgba(20,0,40,0.8); }
            }
            .main-container {
                border: 2px solid #00FFFF !important;
                box-shadow: 0 0 20px #FF00FF !important;
            }
            .stButton>button {
                animation: neonGlow 2s infinite alternate !important;
            }
            @keyframes neonGlow {
                from { box-shadow: 0 0 5px #00FFFF, 0 0 10px #FF00FF; }
                to { box-shadow: 0 0 15px #00FFFF, 0 0 20px #FF00FF; }
            }
            .success-message {
                font-family: 'Orbitron', sans-serif !important;
                animation: textGlow 1s infinite alternate;
            }
            @keyframes textGlow {
                from { text-shadow: 0 0 5px #00FFFF; }
                to { text-shadow: 0 0 10px #FF00FF; }
            }
        """
    },
    "🏰 Medieval Squidward Castle": {
        "emojis": ["🏰", "⚔️", "🛡️", "👑", "🎵"],
        "background": "https://media.giphy.com/media/DBW3BniaWrFo4/giphy.gif",
        "primary_color": "#FFD700",  # Gold
        "secondary_color": "#800020",  # Burgundy
        "font_family": "'UnifrakturMaguntia', cursive",  # Medieval font
        "model": "K-Means",
        "quotes": [
            "Hear ye! Hear ye! By order of King Neptune! 🎺",
            "This model is sharper than Sir Squidward's clarinet! 🎵",
            "To the stock market... AND BEYOND! 🏹"
        ],
        "custom_css": """
            @import url('https://fonts.googleapis.com/css2?family=UnifrakturMaguntia&display=swap');
            .stApp {
                background-color: rgba(139, 69, 19, 0.3) !important;
                background-blend-mode: overlay;
            }
            .main-container {
                border: 4px ridge #FFD700 !important;
                background: linear-gradient(rgba(139, 69, 19, 0.7), rgba(0, 0, 0, 0.8)) !important;
            }
            h1, h2, h3 {
                text-shadow: 2px 2px 4px #800020 !important;
            }
            .stButton>button {
                background: linear-gradient(to bottom, #FFD700, #800020) !important;
                border: 2px solid #FFD700 !important;
            }
            .success-message {
                font-family: 'UnifrakturMaguntia', cursive !important;
                color: #FFD700 !important;
                text-shadow: 2px 2px 4px #800020 !important;
            }
        """
    }
}

def show_welcome_page():
    """Display animated welcome page with SpongeBob characters"""
    st.markdown(f"""
    <style>
        @import url('https://fonts.cdnfonts.com/css/spongebob-font-condensed');
        
        .welcome-container {{
            position: relative;
            width: 100%;
            height: 100vh;
            background: url('{WELCOME_CONFIG["background"]}');
            background-size: cover;
        }}
        
        .title {{
            font-family: 'SpongeBob Font', cursive;
            font-size: 4rem;
            text-align: center;
            color: black;
            text-shadow: 3px 3px 0 yellow, -1px -1px 0 yellow, 1px -1px 0 yellow, -1px 1px 0 yellow;
            padding-top: 2rem;
            animation: bounce 2s infinite;
        }}
        
        .character-grid {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 2rem;
            padding: 4rem;
        }}
        
        .character {{
            animation: float 3s ease-in-out infinite;
            text-align: center;
        }}
        
        .quote-bubble {{
            background: white;
            padding: 1.5rem;
            border-radius: 20px;
            position: relative;
            margin: 1rem;
            font-family: 'SpongeBob Font', cursive;
            box-shadow: 0 0 20px yellow;
        }}
        
        .spongebob-text {{
            color: black !important;
            font-weight: bold;
        }}
        
        @keyframes float {{
            0% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-20px); }}
            100% {{ transform: translateY(0px); }}
        }}
        
        @keyframes bounce {{
            0%, 100% {{ transform: translateY(0); }}
            50% {{ transform: translateY(-20px); }}
        }}
        
        .continue-button {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 100;
        }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='welcome-container'>", unsafe_allow_html=True)
    
    # Add the title
    st.markdown("<h1 class='title'>KRUSTY KRAB FINANCIAL ML</h1>", unsafe_allow_html=True)
    
    cols = st.columns(5)
    for idx, (name, url) in enumerate(WELCOME_CONFIG["characters"].items()):
        with cols[idx]:
            bg_color = 'lightyellow' if name == 'spongebob' else 'pink' if name == 'patrick' else 'lightblue'
            text_class = "spongebob-text" if name == 'spongebob' else ""
            st.markdown(f"""
                <div class='character'>
                    <img src='{url}' width='200'>
                    <div class='quote-bubble' style='background: {bg_color}'>
                        <h3>{WELCOME_CONFIG["quotes"][idx][2]} {WELCOME_CONFIG["quotes"][idx][0]}</h3>
                        <p class='{text_class}'>{WELCOME_CONFIG["quotes"][idx][1]}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # Simple button to continue instead of JavaScript click handler
    if st.button("ENTER THE KRUSTY KRAB! 🍔", key="enter_button"):
        st.session_state.welcome_dismissed = True
        st.rerun()

def apply_theme(theme):
    st.markdown(f"""
    <style>
        @import url('https://fonts.cdnfonts.com/css/spongebob-font-condensed');
        
        .stApp {{
            background-image: url('{theme["background"]}');
            background-size: cover;
            background-attachment: fixed;
        }}
        
        .main-container {{
            background: rgba(0, 0, 0, 0.7);
            border-radius: 15px;
            padding: 2rem;
            margin: 1rem;
            box-shadow: 0 0 20px {theme["primary_color"]};
        }}
        
        h1, h2, h3, h4, h5, h6, .stMarkdown {{
            color: {theme["primary_color"]} !important;
            font-family: {theme["font_family"]} !important;
        }}
        
        .stButton>button {{
            background: {theme["secondary_color"]} !important;
            color: {theme["primary_color"]} !important;
            border: 2px solid {theme["primary_color"]} !important;
            border-radius: 10px !important;
            font-family: {theme["font_family"]} !important;
            font-weight: bold !important;
        }}
        
        .stSelectbox, .stRadio, .stTextInput, .stMultiselect {{
            font-family: {theme["font_family"]} !important;
        }}
        
        .success-message {{
            animation: rainbow 2s infinite;
            text-align: center;
            font-size: 1.5rem;
            margin: 1rem 0;
        }}
        
        {theme.get("custom_css", "")}
        
        @keyframes rainbow {{
            0% {{ color: red; }}
            14% {{ color: orange; }}
            28% {{ color: yellow; }}
            42% {{ color: green; }}
            57% {{ color: blue; }}
            71% {{ color: indigo; }}
            85% {{ color: violet; }}
            100% {{ color: red; }}
        }}
    </style>
    """, unsafe_allow_html=True)

def show_bubbles():
    """Show SpongeBob-style bubbles with animation and sound"""
    # Bubble animation
    bubbles = ""
    for _ in range(20):
        left = np.random.randint(0, 95)
        size = np.random.randint(10, 30)
        duration = np.random.randint(3, 8)
        delay = np.random.random() * 2
        bubbles += f"""
        <div class='bubble' style='
            position: fixed;
            left:{left}vw; 
            width:{size}px; 
            height:{size}px; 
            background: rgba(255,255,255,0.5);
            border-radius: 50%;
            animation: float {duration}s infinite;
            animation-delay:{delay}s;
            pointer-events: none;
        '></div>
        """
    st.markdown(bubbles, unsafe_allow_html=True)
    
    # Bubble animation CSS
    st.markdown("""
    <style>
        @keyframes float {
            0% { transform: translateY(100vh) scale(0.5); opacity: 0; }
            50% { opacity: 0.8; }
            100% { transform: translateY(-100vh) scale(1.5); opacity: 0; }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Audio element with JavaScript to handle playback
    st.markdown("""
    <audio id="bubbleAudio" controls style="display:none">
        <source src="https://assets.mixkit.co/sfx/preview/mixkit-bubble-pop-with-delay-2354.mp3" type="audio/mpeg">
    </audio>
    <script>
        // Play sound when bubbles appear
        setTimeout(() => {
            const audio = document.getElementById("bubbleAudio");
            audio.volume = 0.3; // Lower volume
            audio.play().catch(e => console.log("Audio play prevented:", e));
        }, 500);
    </script>
    """, unsafe_allow_html=True)

def load_data(uploaded_file, ticker):
    """Load data from either uploaded file or Yahoo Finance"""
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.session_state.data_source = "uploaded"
            return df
        except Exception as e:
            st.error(f"Barnacles! Couldn't read that file: {e}")
            return None
    elif ticker:
        try:
            with st.spinner("Flipping Krabby Patties..."):
                stock_data = yf.download(ticker, period="1y")
                if not stock_data.empty:
                    df = stock_data.reset_index()
                    st.session_state.df = df
                    st.session_state.data_source = "yfinance"
                    return df
                else:
                    st.error("Barnacles! No data found for that ticker!")
                    return None
        except Exception as e:
            st.error(f"Tartar sauce! Error fetching data: {e}")
            return None
    return None

def train_model(model_type, X_train, y_train, X_test, y_test):
    """Train the selected model and return predictions"""
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_type == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        return {
            "model": model,
            "y_pred": y_pred,
            "y_test": y_test,
            "metrics": {
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "R²": r2_score(y_test, y_pred)
            }
        }
    elif model_type == "Logistic Regression":
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        return {
            "model": model,
            "y_pred": y_pred,
            "y_test": y_test,
            "metrics": {
                "Accuracy": accuracy_score(y_test, y_pred)
            }
        }
    elif model_type == "K-Means":
        model = KMeans(n_clusters=3, random_state=42)
        clusters = model.fit_predict(X_train_scaled)
        return {
            "model": model,
            "clusters": clusters,
            "X_train": X_train,
            "features": X_train.columns.tolist(),
            "metrics": {
                "Silhouette Score": silhouette_score(X_train_scaled, clusters)
            }
        }
    return None

def main():
    # Initialize session state
    if 'welcome_dismissed' not in st.session_state:
        st.session_state.welcome_dismissed = False
    if 'show_bubbles' not in st.session_state:
        st.session_state.show_bubbles = False
    if 'model_results' not in st.session_state:
        st.session_state.model_results = None
    if 'df' not in st.session_state:
        st.session_state.df = None

    # Show welcome page only once
    if not st.session_state.welcome_dismissed:
        show_welcome_page()
        return

    # Main application logic
    st.sidebar.header("Krusty Krab Financial ML")
    theme_name = st.sidebar.selectbox("CHOOSE YOUR ADVENTURE:", list(THEMES.keys()))
    theme = THEMES[theme_name]
    apply_theme(theme)
    
    # Header with SpongeBob theme
    st.markdown(f"""
    <div class='main-container'>
        <h1 style="text-align:center; color:{theme['primary_color']}; text-shadow: 2px 2px {theme['secondary_color']}">
            {theme['emojis'][0]} KRUSTY KRAB FINANCIAL MACHINE LEARNING {theme['emojis'][0]}
        </h1>
        <h3 style="text-align:center; color:{theme['secondary_color']}">
            "{np.random.choice(theme['quotes'])}"
        </h3>
    </div>
    """, unsafe_allow_html=True)

    # Data Input Section
    st.sidebar.header("🍔 DATA KETCHUP")
    data_source = st.sidebar.radio("CHOOSE YOUR INGREDIENTS:", 
                                 ["📤 Upload Secret Formula", "📈 Fetch Live Krabby Patties"])
    
    uploaded_file = None
    ticker = ""
    if data_source == "📤 Upload Secret Formula":
        uploaded_file = st.sidebar.file_uploader("UPLOAD KRABBY PATTY RECIPE:", type=["csv", "xlsx"])
    else:
        ticker = st.sidebar.text_input("ENTER KRABBY STOCK SYMBOL:", "AAPL").upper()
    
    if st.sidebar.button("LOAD THE CHUM BUCKET!"):
        df = load_data(uploaded_file, ticker)
        if df is not None:
            st.session_state.df = df
            st.sidebar.success("Sizzling fresh patties! 🔥")
            # Show bubbles when data loads successfully
            st.session_state.show_bubbles = True
            st.rerun()

    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        
        with st.expander("👀 PEEK AT THE SECRET FORMULA", expanded=True):
            st.write(df.head().style.background_gradient(cmap='YlGnBu'))
            st.write(f"📊 Dataset shape: {df.shape}")
        
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2:
            st.error("Barnacles! Need more numeric ingredients!")
            return
            
        # Model configuration
        st.header("🔮 Krusty Krab Financial ML Training")
        st.write(f"🧠 Selected Model: **{theme['model']}**")
        
        target = st.selectbox("🎯 TARGET VARIABLE (The Chum Bucket):", numeric_cols)
        features = st.multiselect("📊 FEATURES (Secret Sauce):", numeric_cols, 
                                 default=[c for c in numeric_cols if c != target][:min(3, len(numeric_cols)-1)])
        
        if theme['model'] == "Logistic Regression":
            st.info("🦀 Note: For Logistic Regression, target will be binarized (above/below median)")
        
        if st.button(f"🚀 TRAIN THE {theme['model'].upper()} MODEL!"):
            if not features:
                st.error("Tartar sauce! You need to select at least one feature!")
                return
                
            with st.spinner("Mr. Krabs is counting money..."):
                X = df[features]
                y = df[target]
                
                # Prepare data based on model type
                if theme['model'] == "Logistic Regression":
                    y = (y > y.median()).astype(int)
                    st.warning("🦑 Target converted to binary (0/1) based on median value")
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                results = train_model(theme['model'], X_train, y_train, X_test, y_test)
                if results:
                    st.session_state.model_results = results
                    st.session_state.show_bubbles = True
                    st.success("Krabby Model Ready! Time to make some $$$!")
                    st.rerun()

    # Show results if available
    if st.session_state.get('model_results'):
        results = st.session_state.model_results
        
        st.header("📊 MODEL RESULTS")
        st.markdown("<div class='success-message'>FUTURE PREDICTION SUCCESSFUL!</div>", unsafe_allow_html=True)
        
        # Display metrics
        cols = st.columns(len(results['metrics']))
        for idx, (metric_name, metric_value) in enumerate(results['metrics'].items()):
            with cols[idx]:
                st.metric(metric_name, f"{metric_value:.2f}")
        
        # Show appropriate visualization based on model type
        if theme['model'] in ["Linear Regression", "Logistic Regression"]:
            # Actual vs Predicted plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=results['y_test'],
                mode='markers',
                name='Actual',
                marker=dict(color=theme['primary_color'])
            ))
            fig.add_trace(go.Scatter(
                y=results['y_pred'],
                mode='markers',
                name='Predicted',
                marker=dict(color=theme['secondary_color'])
            ))
            fig.update_layout(
                title="Actual vs Predicted Values",
                xaxis_title="Sample Index",
                yaxis_title="Value"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance if available
            if hasattr(results['model'], 'coef_'):
                importance = pd.DataFrame({
                    'Feature': features,
                    'Importance': results['model'].coef_[0]
                }).sort_values('Importance', key=abs, ascending=False)
                
                fig = px.bar(importance, x='Importance', y='Feature', orientation='h',
                            title="Feature Importance (How much each ingredient matters)")
                st.plotly_chart(fig, use_container_width=True)
        
        elif theme['model'] == "K-Means":
            # Get the necessary data from results
            features = results['features']
            clusters = results['clusters']
            X_train = results['X_train']
            
            # Create a DataFrame for visualization
            viz_df = X_train.copy()
            viz_df['Cluster'] = clusters
            
            if len(features) >= 3:
                fig = px.scatter_3d(
                    viz_df,
                    x=features[0],
                    y=features[1],
                    z=features[2],
                    color='Cluster',
                    title="Krabby Clusters in 3D Space! 🌟",
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.scatter(
                    viz_df,
                    x=features[0],
                    y=features[1] if len(features) > 1 else features[0],
                    color='Cluster',
                    title="Krabby Clusters! 🌟"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Show bubbles if flag is set
    if st.session_state.show_bubbles:
        show_bubbles()
        st.session_state.show_bubbles = False  # Reset after showing

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import plotly.express as px
import plotly.graph_objects as go

# Sayfa Ayarları
st.set_page_config(
    page_title="Alperen Sengun AI Predictor",
    page_icon="🏀",
    layout="wide"
)

# --- 1. VERİ YÜKLEME VE İŞLEME FONKSİYONLARI ---
@st.cache_data
def load_and_prep_data():
    # Veriyi yükle
    try:
        df = pd.read_csv('alperen_sengun_full_career_ml_data.csv')
    except FileNotFoundError:
        st.error("CSV dosyası bulunamadı! 'alperen_sengun_full_career_ml_data.csv' dosyasının aynı klasörde olduğundan emin ol.")
        return None, None, None

    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE').reset_index(drop=True)

    # Savunma Gücü Haritası
    DEFENSE_RATING = {
        'MIN': 95, 'CLE': 92, 'BOS': 85, 'OKC': 82, 'ORL': 85, 'DEN': 80,
        'MIA': 82, 'NYK': 78, 'NOP': 78, 'MEM': 75, 'LAL': 72, 'MIL': 75,
        'PHI': 74, 'DAL': 70, 'LAC': 72, 'IND': 68, 'CHI': 65, 'TOR': 65,
        'GSW': 62, 'PHX': 60, 'ATL': 58, 'SAC': 55, 'UTA': 55, 'BKN': 52,
        'SAS': 58, 'DET': 55, 'CHA': 50, 'POR': 48, 'WAS': 45
    }
    df['OPP_DEF_STRENGTH'] = df['OPPONENT'].map(DEFENSE_RATING).fillna(70)

    # Rolling Metrics
    rolling_metrics = ['PTS', 'REB', 'AST', 'MIN', 'FGA', 'TOV']
    for col in rolling_metrics:
        df[f'L3_{col}'] = df[col].rolling(3).mean().shift(1)
        df[f'L5_{col}'] = df[col].rolling(5).mean().shift(1)
        df[f'L10_{col}'] = df[col].rolling(10).mean().shift(1)

    df['IS_B2B'] = (df['REST_DAYS'] == 0).astype(int)

    # Sezon Ortalamaları
    for col in ['PTS', 'REB', 'FG_PCT']:
        df[f'SZN_{col}'] = df.groupby('SEASON_ID')[col].transform(lambda x: x.expanding().mean().shift(1))

    df_ml = df.dropna().reset_index(drop=True)
    return df, df_ml, DEFENSE_RATING

# --- 2. MODEL EĞİTİMİ ---
@st.cache_resource
def train_model(df_ml):
    target = 'PTS'
    features = [
        'IS_HOME', 'IS_B2B', 'OPP_DEF_STRENGTH', 'REST_DAYS',
        'L3_PTS', 'L5_PTS', 'L10_PTS',
        'L3_MIN', 'L5_MIN',
        'L5_FGA', 'L5_TOV',
        'SZN_PTS', 'SZN_FG_PCT'
    ]

    X = df_ml[features]
    y = df_ml[target]

    # Modeli tüm veriyle eğitiyoruz (Geleceği tahmin etmek için)
    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
    )
    model.fit(X, y)
    
    # Hata payı hesaplama (Güven aralığı için)
    preds = model.predict(X)
    std_dev = np.std(y - preds)
    
    return model, features, std_dev

# --- 3. ANA UYGULAMA ---
def main():
    st.title("🏀 Alperen Sengun: AI Performance Predictor")
    st.markdown("This dashboard predicts Alperen's next game stats using machine learning techniques based on historical data.")

    raw_df, df_ml, defense_map = load_and_prep_data()
    
    if df_ml is not None:
        model, features, std_dev = train_model(df_ml)

        # Yan Menü (Sidebar) - Simülasyon
        st.sidebar.header("🛠️ Prediction Simulator")
        st.sidebar.markdown("Change the conditions to see how the prediction changes.")
        
        # Seçimler
        teams = sorted(defense_map.keys())
        # Default index olarak BKN yoksa listenin başını al
        default_index = teams.index('BKN') if 'BKN' in teams else 0
        selected_opp = st.sidebar.selectbox("Opponent Team", options=teams, index=default_index)
        
        # DÜZELTİLDİ: Türkçe/İngilizce uyumsuzluğu giderildi
        is_home_label = st.sidebar.radio("Venue", ["Away", "Home"])
        is_home_val = 1 if is_home_label == "Home" else 0
        
        rest_days = st.sidebar.slider("Rest Days", 0, 10, 7)
        
        # Simülasyon Verisini Hazırla (Son maçın form durumunu alıyoruz)
        # last_game_idx = raw_df.index[-1] # Kullanılmıyor, sildim
        
        # Son maçın istatistiklerine göre rolling average hesapla
        l3_pts = raw_df['PTS'].tail(3).mean()
        l5_pts = raw_df['PTS'].tail(5).mean()
        l10_pts = raw_df['PTS'].tail(10).mean()
        l3_min = raw_df['MIN'].tail(3).mean()
        l5_min = raw_df['MIN'].tail(5).mean()
        l5_fga = raw_df['FGA'].tail(5).mean()
        l5_tov = raw_df['TOV'].tail(5).mean()
        
        # Sezon verilerini son maçın sezon ID'sine göre çek
        last_season_id = raw_df.iloc[-1]['SEASON_ID']
        szn_pts = raw_df[raw_df['SEASON_ID'] == last_season_id]['PTS'].mean()
        szn_fg = raw_df[raw_df['SEASON_ID'] == last_season_id]['FG_PCT'].mean()

        input_data = pd.DataFrame({
            'IS_HOME': [is_home_val],
            'IS_B2B': [1 if rest_days == 0 else 0],
            'OPP_DEF_STRENGTH': [defense_map[selected_opp]],
            'REST_DAYS': [rest_days],
            'L3_PTS': [l3_pts], 'L5_PTS': [l5_pts], 'L10_PTS': [l10_pts],
            'L3_MIN': [l3_min], 'L5_MIN': [l5_min],
            'L5_FGA': [l5_fga], 'L5_TOV': [l5_tov],
            'SZN_PTS': [szn_pts], 'SZN_FG_PCT': [szn_fg]
        })

        # Tahmin Yap
        prediction = model.predict(input_data)[0]
        
        # --- ANA EKRAN GÖRÜNÜMÜ ---
        
        # 1. Kısım: Tahmin Kartı
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader(f"🆚 {selected_opp}")
            st.caption(f"{is_home_label}, {rest_days} days rest.")
            
        with col2:
            st.metric(
                label="Predicted Score", 
                value=f"{prediction:.1f}",
                delta=f"Diff from last game: {prediction - raw_df.iloc[-1]['PTS']:.1f}"
            )
            
        with col3:
            low = prediction - std_dev
            high = prediction + std_dev
            st.info(f"🎯 Confidence Interval: **{low:.1f} - {high:.1f}**")

        st.divider()

        # 2. Kısım: Grafiksel Analiz
        col_chart1, col_chart2 = st.columns([2, 1])

        with col_chart1:
            st.subheader("📈 Prediction vs Actual (Last 20 Games)")
            # Son 20 maçın tahminlerini oluştur
            recent_data = df_ml.tail(20).copy()
            recent_data['Tahmin'] = model.predict(recent_data[features])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=recent_data['GAME_DATE'], y=recent_data['PTS'], mode='lines+markers', name='Actual', line=dict(color='#FF4B4B')))
            fig.add_trace(go.Scatter(x=recent_data['GAME_DATE'], y=recent_data['Tahmin'], mode='lines+markers', name='Predicted', line=dict(color='#1F77B4', dash='dot')))
            
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col_chart2:
            st.subheader("🔑 Key Features")
            importance = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True).tail(10)
            
            fig_imp = px.bar(importance, x='Importance', y='Feature', orientation='h', color='Importance')
            fig_imp.update_layout(height=350, showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_imp, use_container_width=True)

        # 3. Kısım: Son Maçlar Tablosu
        with st.expander("📋 Last 10 Games Stats"):
            st.dataframe(raw_df[['GAME_DATE', 'OPPONENT', 'PTS', 'REB', 'AST', 'MIN']].tail(10).sort_values('GAME_DATE', ascending=False))

if __name__ == "__main__":
    main()

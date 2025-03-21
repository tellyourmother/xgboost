import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# ------------------------------
# Helper: Get current NBA season
# ------------------------------
def get_current_season():
    today = datetime.today()
    year = today.year
    if today.month >= 10:  # NBA season starts in October
        return f"{year}-{str(year + 1)[-2:]}"
    else:
        return f"{year - 1}-{str(year)[-2:]}"


# ------------------------------
# App Layout
# ------------------------------
st.title("🏀 NBA Player Stat Predictor")
st.markdown("Predict a player's next game stats and compare performance to their season average.")

# Input: Player name
player_name = st.text_input("Enter the full name of an NBA player (e.g., LeBron James):")

# Input: Stat to compare to season average
stat_to_check = st.selectbox("Stat to compare against season average:", ['PTS', 'REB', 'AST', 'FG_PCT'])

if player_name:
    # Step 1: Find player
    nba_players = players.get_players()
    player_dict = next((p for p in nba_players if p['full_name'].lower() == player_name.lower()), None)

    if player_dict:
        player_id = player_dict['id']
        season = get_current_season()
        st.info(f"Fetching data for **{player_name}** in the **{season}** season...")

        # Step 2: Get game log
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star='Regular Season'
        )
        df = gamelog.get_data_frames()[0]

        if len(df) < 5:
            st.error("❌ Not enough games this season to build a prediction model.")
        else:
            # ✅ Prepare data
            df = df.sort_values('GAME_DATE').reset_index(drop=True)

            # Features and Targets
            target_cols = ['PTS', 'REB', 'AST', 'FG_PCT']
            feature_cols = ['MIN', 'FGA', 'FGM', 'FG3A', 'FG3M', 'FTA', 'FTM',
                            'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF']

            feature_df = df[feature_cols + target_cols].shift(1).iloc[1:]
            target_df = df[target_cols].iloc[1:]

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(feature_df)
            y = target_df.values

            # Train model
            xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3)
            model = MultiOutputRegressor(xgb)
            model.fit(X_scaled, y)

            # Predict next game
            latest_features = df[feature_cols + target_cols].iloc[-1:].values
            latest_scaled = scaler.transform(latest_features)
            prediction = model.predict(latest_scaled)
            pred_df = pd.DataFrame(prediction, columns=target_cols)

            st.subheader("🔮 Predicted Next Game Stats:")
            st.dataframe(pred_df.round(2), use_container_width=True)

            # Last 15 games overview
            recent_df = df.sort_values("GAME_DATE").tail(15).copy()

            # Compare selected stat to season average
            season_avg = df[stat_to_check].mean()
            games_over_avg = df[df[stat_to_check] > season_avg]
            count_over = len(games_over_avg)

            st.markdown(f"### 📊 {player_name}'s Performance in Last 15 Games")
            st.write(f"**Season average {stat_to_check}:** {season_avg:.2f}")
            st.write(f"**Games above season average:** {count_over} out of {len(df)}")

            st.markdown("#### 🗓️ Games Over Season Average:")
            st.dataframe(games_over_avg[['GAME_DATE', 'MATCHUP', stat_to_check]])

            st.markdown("#### 📋 Full Game Log (Last 15 Games):")
            st.dataframe(recent_df[['GAME_DATE', 'MATCHUP'] + target_cols].sort_values('GAME_DATE'), use_container_width=True)

    else:
        st.error(f"❌ Player '{player_name}' not found.")

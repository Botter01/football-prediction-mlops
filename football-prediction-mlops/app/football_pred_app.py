import streamlit as st
import pandas as pd
import requests
import json
import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from io import StringIO

load_dotenv()

SCORING_URI = os.getenv("SCORING_URI")
ENDPOINT_KEY = os.getenv("ENDPOINT_KEY")
STORAGE_ACCOUNT_NAME = os.getenv("STORAGE_ACCOUNT_NAME")
STORAGE_ACCOUNT_KEY = os.getenv("STORAGE_ACCOUNT_KEY")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")

st.set_page_config(page_title="Premier League Predikció", page_icon="⚽")

@st.cache_data(ttl=3600)
def load_historical_data():
    blob_service = BlobServiceClient(
        account_url=f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
        credential=STORAGE_ACCOUNT_KEY
    )
    blob_client = blob_service.get_blob_client(
        container=CONTAINER_NAME,
        blob="football-data/raw/premier_league_raw.csv"
    )
    data = blob_client.download_blob().readall()
    df = pd.read_csv(StringIO(data.decode("utf-8")))
    return df

def get_points(row, team):
    if row["HomeTeam"] == team:
        if row["FTR"] == "H": return 3
        elif row["FTR"] == "D": return 1
        else: return 0
    elif row["AwayTeam"] == team:
        if row["FTR"] == "A": return 3
        elif row["FTR"] == "D": return 1
        else: return 0
    return None

def get_goal_diff(row, team):
    if row["HomeTeam"] == team:
        return row["FTHG"] - row["FTAG"]
    elif row["AwayTeam"] == team:
        return row["FTAG"] - row["FTHG"]
    return None

def get_team_features(home_team, away_team, historical_df, n=5):
    df = historical_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    home_history = df[(df["HomeTeam"] == home_team) | (df["AwayTeam"] == home_team)].tail(n)
    away_history = df[(df["HomeTeam"] == away_team) | (df["AwayTeam"] == away_team)].tail(n)

    if len(home_history) == 0 or len(away_history) == 0:
        raise ValueError("Nincs elég korábbi meccs-adat ehhez a csapathoz.")

    home_form = home_history.apply(lambda r: get_points(r, home_team), axis=1).mean()
    away_form = away_history.apply(lambda r: get_points(r, away_team), axis=1).mean()
    home_goal_diff_trend = home_history.apply(lambda r: get_goal_diff(r, home_team), axis=1).mean()
    away_goal_diff_trend = away_history.apply(lambda r: get_goal_diff(r, away_team), axis=1).mean()

    h2h = df[
        ((df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team)) |
        ((df["HomeTeam"] == away_team) & (df["AwayTeam"] == home_team))
    ].tail(n)

    if len(h2h) > 0:
        h2h_home_points = h2h.apply(lambda r: get_points(r, home_team), axis=1).mean()
    else:
        h2h_home_points = 1

    return {
        "home_form": home_form,
        "away_form": away_form,
        "home_goal_diff_trend": home_goal_diff_trend,
        "away_goal_diff_trend": away_goal_diff_trend,
        "h2h_home_points": h2h_home_points
    }

def call_prediction_endpoint(features):
    payload = {
        "input_data": {
            "columns": ["home_form", "away_form", "home_goal_diff_trend", "away_goal_diff_trend", "h2h_home_points"],
            "data": [[
                features["home_form"], features["away_form"],
                features["home_goal_diff_trend"], features["away_goal_diff_trend"],
                features["h2h_home_points"]
            ]]
        }
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ENDPOINT_KEY}"
    }
    response = requests.post(SCORING_URI, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    return response.json()

st.title("⚽ Premier League Meccs Predikció")
st.write("Azure ML-lel tanított modell, ami a csapatok formája és statisztikái alapján jósolja a kimenetelt.")

df = load_historical_data()
teams = sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Hazai csapat", teams, index=teams.index("Liverpool") if "Liverpool" in teams else 0)
with col2:
    away_team = st.selectbox("Vendég csapat", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 1)

if st.button("Predikció", type="primary"):
    if home_team == away_team:
        st.error("Válassz két különböző csapatot!")
    else:
        with st.spinner("Feature-ök számítása és predikció lekérése..."):
            try:
                features = get_team_features(home_team, away_team, df)
                result = call_prediction_endpoint(features)

                label_map = {0: "Hazai győzelem", 1: "Döntetlen", 2: "Vendég győzelem"}
                prediction = result[0] if isinstance(result, list) else result

                st.success(f"**Predikció: {label_map.get(prediction, prediction)}**")

                with st.expander("Számított feature-ök megtekintése"):
                    st.json(features)

            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Hiba történt: {e}")
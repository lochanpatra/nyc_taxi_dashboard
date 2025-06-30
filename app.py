import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings

# --- Page Setup ---
st.set_page_config(layout="wide")
st.title("🚖 NYC Yellow Taxi Trip Analysis Dashboard")

# --- Load Data ---
@st.cache_data
def load_trip_data(folder_path):
    all_files = sorted(glob.glob(f"{folder_path}/yellow_tripdata_part_*.csv"))
    df_list = []
    for file in all_files:
        temp_df = pd.read_csv(
            file,
            parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"],
            low_memory=False
        )
        df_list.append(temp_df)
    df = pd.concat(df_list, ignore_index=True)

    # Add derived columns
    df["pickup_date"] = df["tpep_pickup_datetime"].dt.date
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["trip_duration_minutes"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    return df

@st.cache_data
def load_zone_lookup(path):
    return pd.read_csv(path)

# --- File Paths ---
trip_data_folder = "data"
zone_lookup_path = "data/taxi_zone_lookup.csv"

# --- Load & Filter Data ---
with st.spinner("Loading data..."):
    df = load_trip_data(trip_data_folder)
    zones = load_zone_lookup(zone_lookup_path)

# Optional: downsample for performance
if st.sidebar.checkbox("🔽 Sample 10% of data (for speed)", value=True):
    df = df.sample(frac=0.1, random_state=42)

# --- Sidebar Filters ---
st.sidebar.header("🛠️ Filters")
start_date = st.sidebar.date_input("Start Date", datetime(2024, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2024, 1, 31))

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()

df = df[(df["pickup_date"] >= start_date) & (df["pickup_date"] <= end_date)]

payment_options = df["payment_type"].dropna().unique()
payment_selected = st.sidebar.selectbox("Payment Type", payment_options)
df = df[df["payment_type"] == payment_selected]

# --- Clean Data ---
df_clean = df[
    (df["passenger_count"] > 0) &
    (df["trip_distance"] > 0) &
    (df["trip_distance"] < 100) &
    (df["fare_amount"] > 0)
].copy()

# --- Merge Zone Info ---
df_clean = df_clean.merge(zones, left_on="PULocationID", right_on="LocationID", how="left", suffixes=("", "_PU"))
df_clean = df_clean.merge(zones, left_on="DOLocationID", right_on="LocationID", how="left", suffixes=("", "_DO"))

# --- Zone Aggregation ---
st.subheader("📌 Top Pickup Zones")
pickup_counts = df_clean["Zone"].value_counts().head(10).reset_index()
pickup_counts.columns = ["Zone", "Trip Count"]
st.dataframe(pickup_counts)

# --- Charts ---
st.subheader("📊 Trips per Hour")
hourly_trips = df_clean.groupby("pickup_hour").size().reset_index(name="Trip Count")
fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.barplot(data=hourly_trips, x="pickup_hour", y="Trip Count", hue="pickup_hour", palette="viridis", legend=False, ax=ax1)
ax1.set_title("Trips by Hour")
st.pyplot(fig1)

st.subheader("📈 Daily Total Fare")
daily_fares = df_clean.groupby("pickup_date")["fare_amount"].sum().reset_index()
fig2, ax2 = plt.subplots(figsize=(12, 4))
sns.lineplot(data=daily_fares, x="pickup_date", y="fare_amount", marker="o", ax=ax2)
ax2.set_title("Daily Fare Total")
fig2.autofmt_xdate()
st.pyplot(fig2)

st.subheader("💳 Payment Type Distribution")
pay_dist = df_clean["payment_type"].value_counts().reset_index()
pay_dist.columns = ["Payment Type", "Trip Count"]
fig3, ax3 = plt.subplots()
sns.barplot(data=pay_dist, x="Payment Type", y="Trip Count", hue="Payment Type", palette="pastel", legend=False, ax=ax3)
ax3.set_title("Payment Type Counts")
st.pyplot(fig3)

st.subheader("📉 Distance vs Fare")
df_scatter = df_clean[(df_clean["trip_distance"] < 50) & (df_clean["fare_amount"] < 200)]
fig4, ax4 = plt.subplots(figsize=(10, 5))
sns.scatterplot(
    data=df_scatter.sample(frac=0.02, random_state=1),
    x="trip_distance",
    y="fare_amount",
    alpha=0.3,
    ax=ax4
)
ax4.set_title("Trip Distance vs Fare")
st.pyplot(fig4)

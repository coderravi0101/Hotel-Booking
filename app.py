# -----------------------------------------------------------------------------
# Created by: Ravi Kumar Singh
# Executive Hotel Analytics Dashboard (Streamlit)
# Repository: https://github.com/coderravi0101/Hotel-Booking
# -----------------------------------------------------------------------------

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ---------------- 1. Page Configuration ----------------
st.set_page_config(
    page_title="Executive Hotel Analytics Dashboard",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Production-ready config (env override for data path)
DEFAULT_DATA_PATH = os.getenv(
    "HOTEL_DATA_PATH",
    r"D:\\Hotel Booking EDA Machine Learning Project\\Hotel Bookings.csv",
)

# ---------------- Styling: Full-page background + readable content ----------------
BACKGROUND_IMAGE = (
    "https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?w=1600&auto=format&fit=crop"
)

st.markdown(f"""
    <style>
    /* Full page background image */
    .stApp {{
        background-image: url('{BACKGROUND_IMAGE}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    /* Make main content slightly translucent for readability */
    .stApp .main {{
        background-color: rgba(255,255,255,0.88) !important;
        border-radius: 12px;
        padding: 1rem 1rem 2rem 1rem;
    }}
    /* Metric card look */
    .stMetric {{
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
    }}
    /* Hero text styling */
    .hero-text h1 {{
        margin: 0; font-weight: 800; letter-spacing: 1px;
    }}
    </style>
""", unsafe_allow_html=True)

# ---------------- 2. Data Loading & Cleaning (robust & cached) ----------------
@st.cache_data(show_spinner=True)
def load_data(path_or_buffer):
    # Accept either a path string/Path or an uploaded file-like object
    df = pd.read_csv(path_or_buffer)

    # Normalize columns and fill missing
    for col, fill in [
        ("children", 0),
        ("country", "Unknown"),
        ("agent", 0),
        ("company", 0),
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(fill)
        else:
            df[col] = fill

    # Reservation date handling (if present)
    if "reservation_status_date" in df.columns:
        df["reservation_status_date"] = pd.to_datetime(
            df["reservation_status_date"], errors="coerce"
        )

    # Month ordering
    if "arrival_date_month" in df.columns:
        month_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        df["arrival_date_month"] = pd.Categorical(
            df["arrival_date_month"], categories=month_order, ordered=True
        )

    # Feature engineering
    df["stays_in_weekend_nights"] = df.get("stays_in_weekend_nights", 0).fillna(0)
    df["stays_in_week_nights"] = df.get("stays_in_week_nights", 0).fillna(0)
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]

    df["adults"] = df.get("adults", 0).fillna(0)
    df["children"] = df.get("children", 0).fillna(0)
    df["babies"] = df.get("babies", 0).fillna(0)
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]

    # Ensure numeric adr & lead_time
    df["adr"] = pd.to_numeric(df.get("adr", 0), errors="coerce").fillna(0)
    df["lead_time"] = pd.to_numeric(df.get("lead_time", 0), errors="coerce").fillna(0)

    df["is_canceled"] = pd.to_numeric(df.get("is_canceled", 0), errors="coerce").fillna(0).astype(int)

    df["estimated_revenue"] = df.apply(
        lambda r: r.get("adr", 0) * r.get("total_nights", 0) if r.get("is_canceled", 0) == 0 else 0,
        axis=1,
    )

    # Useful derived fields
    if all(c in df.columns for c in ["arrival_date_year", "arrival_date_month"]):
        # Try to compute a synthetic arrival_date for richer EDA (day may be missing)
        if "arrival_date_day_of_month" in df.columns:
            try:
                df["arrival_date"] = pd.to_datetime(
                    df["arrival_date_year"].astype(str)
                    + "-"
                    + df["arrival_date_month"].astype(str)
                    + "-"
                    + df["arrival_date_day_of_month"].astype(str),
                    errors="coerce",
                )
                df["arrival_weekday"] = df["arrival_date"].dt.day_name()
            except Exception:
                df["arrival_weekday"] = None
        else:
            df["arrival_weekday"] = None

    return df

# Load data with fallback to uploader
try:
    data_path = Path(DEFAULT_DATA_PATH)
    if data_path.exists():
        df = load_data(data_path)
    else:
        st.sidebar.info("Default dataset not found. Upload 'Hotel Bookings.csv' or set HOTEL_DATA_PATH env variable.")
        uploaded = st.sidebar.file_uploader("Upload Hotel Bookings CSV", type=["csv"])
        if uploaded is None:
            st.sidebar.warning("Please upload the CSV to continue.")
            st.stop()
        df = load_data(uploaded)
except Exception as e:
    st.error(f"⚠️ Failed to load dataset: {e}")
    st.stop()

# ---------------- Sidebar Filters (kept compact) ----------------
st.sidebar.title("🎛️ Executive Filters")

hotel_options = ["All"] + sorted(df["hotel"].dropna().unique().tolist())
hotel_filter = st.sidebar.selectbox("Hotel Type", hotel_options)

year_options = ["All"] + sorted(df["arrival_date_year"].dropna().unique().tolist())
year_filter = st.sidebar.selectbox("Arrival Year", year_options)

segment_options = ["All"] + sorted(df["market_segment"].dropna().astype(str).unique().tolist())
segment_filter = st.sidebar.selectbox("Market Segment", segment_options)

customer_options = ["All"] + sorted(df["customer_type"].dropna().astype(str).unique().tolist())
customer_filter = st.sidebar.selectbox("Customer Type", customer_options)

cancel_options = ["All", "Confirmed", "Cancelled"]
cancel_filter = st.sidebar.selectbox("Booking Status", cancel_options)

# Apply filters
filtered = df.copy()
if hotel_filter != "All":
    filtered = filtered[filtered["hotel"] == hotel_filter]
if year_filter != "All":
    filtered = filtered[filtered["arrival_date_year"] == year_filter]
if segment_filter != "All":
    filtered = filtered[filtered["market_segment"].astype(str) == segment_filter]
if customer_filter != "All":
    filtered = filtered[filtered["customer_type"].astype(str) == customer_filter]
if cancel_filter == "Cancelled":
    filtered = filtered[filtered["is_canceled"] == 1]
elif cancel_filter == "Confirmed":
    filtered = filtered[filtered["is_canceled"] == 0]

# ---------------- Main content ----------------
st.markdown("""
    <div style='text-align:center; padding-top:18px;'>
        <div class='hero-text'>
            <h1>HOTEL BOOKING ANALYTICS</h1>
            <p style='color:#666; margin-top:6px;'>Strategic Performance & Revenue Dashboard</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Key KPIs
total_bookings = len(filtered)
avg_adr = filtered[filtered["is_canceled"] == 0]["adr"].mean() if total_bookings else 0
total_revenue = filtered["estimated_revenue"].sum()
avg_stay = filtered["total_nights"].mean() if total_bookings else 0
cancel_rate = filtered["is_canceled"].mean() * 100 if total_bookings else 0

c1, c2, c3, c4, c5 = st.columns([1.2,1.2,1.2,1.2,1.2])
c1.metric("Total Bookings", f"{total_bookings:,}")
c2.metric("Total Revenue", f"${total_revenue:,.0f}")
c3.metric("Average ADR", f"${avg_adr:,.2f}")
c4.metric("Cancellation Rate", f"{cancel_rate:.1f}%")
c5.metric("Avg Length of Stay", f"{avg_stay:.1f} Nights")

st.markdown("---")

# --------------- Tabs with expanded EDA & production-ready charts ---------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Demand & Trends",
    "🛡️ Cancellation Risk",
    "🌐 Geography & Demographic",
    "💵 Revenue & Pricing",
    "��� Data Explorer"
])

with tab1:
    st.subheader("Booking Demand & Seasonality")
    col1, col2 = st.columns(2)

    with col1:
        # Monthly seasonality (counts) + stacked by year
        monthly = filtered.groupby(["arrival_date_month", "arrival_date_year"], observed=False).size().reset_index(name="bookings")
        fig = px.bar(
            monthly, x="arrival_date_month", y="bookings", color="arrival_date_year",
            title="Monthly Bookings by Year", barmode="group", category_orders={"arrival_date_month": list(filtered["arrival_date_month"].cat.categories) if "arrival_date_month" in filtered.columns el[...]
        )
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap arrival month vs year
        heat = monthly.pivot(index="arrival_date_year", columns="arrival_date_month", values="bookings").fillna(0)
        fig2 = px.imshow(heat, labels=dict(x="Month", y="Year", color="Bookings"), aspect="auto", title="Bookings Heatmap (Year vs Month)")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Bookings by day-of-week if available
        if "arrival_weekday" in filtered.columns and filtered["arrival_weekday"].notna().any():
            dow = filtered["arrival_weekday"].value_counts().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).fillna(0).reset_index()
            dow.columns = ["weekday","bookings"]
            fig_dow = px.bar(dow, x="weekday", y="bookings", title="Bookings by Arrival Weekday")
            st.plotly_chart(fig_dow, use_container_width=True)
        else:
            st.info("Arrival weekday not available in dataset to show weekday-level trend.")

        # Rolling 30-day trend if an arrival_date exists
        if "arrival_date" in filtered.columns and filtered["arrival_date"].notna().any():
            ts = filtered.dropna(subset=["arrival_date"]).set_index("arrival_date").resample("D").size().rename("bookings").to_frame()
            ts["rolling_30"] = ts["bookings"].rolling(30, min_periods=1).mean()
            fig_ts = px.line(ts.reset_index(), x="arrival_date", y=["bookings","rolling_30"], labels={"value":"bookings","arrival_date":"Date"}, title="Daily Bookings + 30-day Rolling Average")
            st.plotly_chart(fig_ts, use_container_width=True)

with tab2:
    st.subheader("Cancellation & Risk Analysis")
    col1, col2 = st.columns(2)
    with col1:
        cancel_by_hotel = filtered.groupby("hotel")["is_canceled"].mean().mul(100).reset_index().sort_values(by="is_canceled", ascending=False)
        fig_c = px.bar(cancel_by_hotel, x="hotel", y="is_canceled", title="Cancellation Rate by Hotel (%)", labels={"is_canceled":"Cancellation %"})
        st.plotly_chart(fig_c, use_container_width=True)

        # ADR distribution for cancelled vs confirmed
        fig_box = px.box(filtered, x="is_canceled", y="adr", points="outliers", labels={"is_canceled":"Cancelled","adr":"ADR"}, title="ADR: Cancelled vs Confirmed")
        st.plotly_chart(fig_box, use_container_width=True)

    with col2:
        # Lead time effect
        fig_sc = px.scatter(filtered.sample(min(len(filtered), 3000)), x="lead_time", y="adr", color="is_canceled", opacity=0.7, title="Lead Time vs ADR (sample)" )
        st.plotly_chart(fig_sc, use_container_width=True)

        # Cancellation over time if arrival_date exists
        if "arrival_date" in filtered.columns and filtered["arrival_date"].notna().any():
            cancel_ts = filtered.copy()
            # Ensure arrival_date is proper datetime values
            cancel_ts["arrival_date"] = pd.to_datetime(cancel_ts["arrival_date"], errors="coerce")

            # If there are no valid datetimes after coercion, show info and skip plotting
            if cancel_ts["arrival_date"].dropna().empty:
                st.info("No valid arrival_date values to compute cancellation rate over time.")
            else:
                try:
                    # Preferred: use resample on a datetime index which is robust across pandas versions
                    cancel_rate_ts = (
                        cancel_ts.dropna(subset=["arrival_date"]) 
                        .set_index("arrival_date")
                        .resample("M")["is_canceled"]
                        .mean()
                        .mul(100)
                        .rename("cancel_rate")
                        .reset_index()
                    )
                except Exception:
                    # Fallback: group by year-month via dt.to_period (avoids pd.Grouper/freq parsing differences)
                    cancel_rate_ts = (
                        cancel_ts.dropna(subset=["arrival_date"]).assign(
                            arrival_month=cancel_ts["arrival_date"].dt.to_period("M").dt.to_timestamp()
                        )
                        .groupby("arrival_month")["is_canceled"].mean().mul(100).rename("cancel_rate").reset_index()
                    )
                    cancel_rate_ts = cancel_rate_ts.rename(columns={"arrival_month":"arrival_date"})

                fig_ct = px.line(cancel_rate_ts, x="arrival_date", y="cancel_rate", title="Monthly Cancellation Rate (%)")
                st.plotly_chart(fig_ct, use_container_width=True)

with tab3:
    st.subheader("Geography & Demographics")
    col1, col2 = st.columns(2)
    with col1:
        country_df = filtered["country"].value_counts().reset_index()
        country_df.columns = ["country","bookings"]
        fig_map = px.choropleth(country_df, locations="country", locationmode='country names', color="bookings", hover_name="country", color_continuous_scale="Viridis", title="Global Booking Density")
        st.plotly_chart(fig_map, use_container_width=True)

    with col2:
        cust_df = filtered["customer_type"].value_counts().reset_index()
        cust_df.columns = ["customer_type","bookings"]
        fig_cust = px.pie(cust_df, names="customer_type", values="bookings", title="Customer Type Split", hole=0.4)
        st.plotly_chart(fig_cust, use_container_width=True)

    # Top origin countries
    top_countries = country_df.head(10).sort_values(by="bookings")
    fig_top = px.bar(top_countries, x="bookings", y="country", orientation="h", title="Top 10 Guest Origin Countries")
    st.plotly_chart(fig_top, use_container_width=True)

with tab4:
    st.subheader("Revenue & Pricing Insights")
    col1, col2 = st.columns(2)
    with col1:
        # ADR trend by month
        adr_month = filtered.groupby("arrival_date_month", observed=False)["adr"].mean().reset_index()
        fig_adr = px.line(adr_month, x="arrival_date_month", y="adr", title="Average ADR by Month")
        st.plotly_chart(fig_adr, use_container_width=True)

        # ADR vs lead time with cancellation coloring
        fig_sc2 = px.scatter(filtered.sample(min(len(filtered), 5000)), x="lead_time", y="adr", color="is_canceled", marginal_x="histogram", marginal_y="box", title="ADR vs Lead Time")
        st.plotly_chart(fig_sc2, use_container_width=True)

    with col2:
        # Length of stay distribution
        fig_stay = px.histogram(filtered[filtered["total_nights"] <= 30], x="total_nights", nbins=30, title="Length of Stay Distribution (<=30 nights)")
        st.plotly_chart(fig_stay, use_container_width=True)

        # Correlation heatmap of numeric features
        numeric_cols = [c for c in ["adr","lead_time","total_nights","total_guests","estimated_revenue"] if c in filtered.columns]
        if len(numeric_cols) >= 2:
            corr = filtered[numeric_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
            st.plotly_chart(fig_corr, use_container_width=True)

with tab5:
    st.subheader("Enterprise Data Explorer")
    display_cols = [
        c for c in [
            "hotel","is_canceled","lead_time","arrival_date_year","arrival_date_month",
            "stays_in_weekend_nights","stays_in_week_nights","adults","children","country",
            "market_segment","customer_type","adr","estimated_revenue","total_nights"
        ] if c in filtered.columns
    ]

    st.dataframe(filtered[display_cols], use_container_width=True, height=500)

    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(label="📥 Download Filtered Dataset (CSV)", data=csv_data, file_name="Executive_Hotel_Data.csv", mime="text/csv")

# ---------------- Footer & Production notes ----------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:12px;'>"
    "Enterprise Hotel Revenue Analytics System • Built with Streamlit & Plotly • Prod-Ready UX • Created by Ravi Kumar Singh"
    "</div>", unsafe_allow_html=True
)

# Helpful tip for deployments
st.sidebar.markdown("""
**Created by:** **Ravi Kumar Singh**  

**Deployment tips:**
- Set HOTEL_DATA_PATH env var on the server or upload the CSV in the UI.
- Use a requirements.txt and pin package versions for reproducible deploys.
- For large datasets, consider pre-processing & storing parquet files.
""")

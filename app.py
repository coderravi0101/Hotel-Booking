import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ---------------- 1. Page Configuration ----------------
st.set_page_config(
    page_title="Executive Hotel Analytics Dashboard",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Professional Styling & Metric Cards
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    .hero-container {
        position: relative;
        text-align: center;
        color: white;
        margin-bottom: 25px;
    }
    .hero-image {
        width: 100%;
        height: 200px;
        object-fit: cover;
        border-radius: 12px;
        filter: brightness(0.65);
    }
    .hero-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-weight: 700;
        letter-spacing: 1px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- 2. Data Loading & Cleaning ----------------
DATA_PATH = Path(__file__).parent / "Hotel Bookings.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    
    # Missing value handling
    df["children"] = df.get("children", pd.Series(0)).fillna(0)
    df["country"] = df.get("country", pd.Series("Unknown")).fillna("Unknown")
    df["agent"] = df.get("agent", pd.Series(0)).fillna(0)
    df["company"] = df.get("company", pd.Series(0)).fillna(0)

    # Date handling
    if "reservation_status_date" in df.columns:
        df["reservation_status_date"] = pd.to_datetime(
            df["reservation_status_date"], errors="coerce"
        )

    # Ordered Categorical Month
    if "arrival_date_month" in df.columns:
        month_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        df["arrival_date_month"] = pd.Categorical(
            df["arrival_date_month"], categories=month_order, ordered=True
        )

    # Feature Engineering
    df["total_nights"] = (
        df.get("stays_in_weekend_nights", 0).fillna(0)
        + df.get("stays_in_week_nights", 0).fillna(0)
    )
    df["total_guests"] = (
        df.get("adults", 0).fillna(0)
        + df.get("children", 0).fillna(0)
        + df.get("babies", 0).fillna(0)
    )
    # Revenue calculations for confirmed bookings only
    df["estimated_revenue"] = df.apply(
        lambda row: row["adr"] * row["total_nights"] if row["is_canceled"] == 0 else 0, 
        axis=1
    )
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"⚠️ Failed to load dataset: {e}")
    st.stop()

# ---------------- 3. Sidebar Filters ----------------
st.sidebar.image("https://images.unsplash.com/photo-1566073771259-6a8506099945?w=500&auto=format&fit=crop", use_container_width=True)
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

# Filter Data logic
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

# ---------------- 4. Hero Header with Image ----------------
st.markdown("""
    <div class="hero-container">
        <img class="hero-image" src="https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?w=1200&auto=format&fit=crop" />
        <div class="hero-text">
            <h1 style="color:white; margin:0;">HOTEL BOOKING ANALYTICS</h1>
            <p style="color:#e0e0e0; font-size: 1.1rem;">Strategic Performance & Revenue Dashboard</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# ---------------- 5. Enterprise KPIs ----------------
total_bookings = len(filtered)
cancel_rate = filtered["is_canceled"].mean() * 100 if total_bookings else 0
avg_adr = filtered[filtered["is_canceled"] == 0]["adr"].mean() if total_bookings else 0
total_revenue = filtered["estimated_revenue"].sum()
avg_stay = filtered["total_nights"].mean() if total_bookings else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Bookings", f"{total_bookings:,}")
c2.metric("Total Revenue", f"${total_revenue:,.0f}")
c3.metric("Average ADR", f"${avg_adr:,.2f}")
c4.metric("Cancellation Rate", f"{cancel_rate:.1f}%")
c5.metric("Avg Length of Stay", f"{avg_stay:.1f} Nights")

st.markdown("---")

# ---------------- 6. Dynamic Analytics Tabs ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Demand & Trends", 
    "🛡️ Cancellation Risk", 
    "🌐 Geography & Demographic", 
    "💵 Revenue Dynamics", 
    "🔍 Enterprise Data Explorer"
])

# COLOR PALETTES FOR PROFESSIONAL LOOK
COLOR_PRIMARY = "#2b5c8f"
COLOR_SECONDARY = "#d9534f"

with tab1:
    st.subheader("Booking Demand Trends")
    col1, col2 = st.columns(2)

    with col1:
        monthly_trend = (
            filtered.groupby(["arrival_date_month"], observed=False)
            .size()
            .reset_index(name="bookings")
        )
        fig_month = px.line(
            monthly_trend,
            x="arrival_date_month",
            y="bookings",
            markers=True,
            title="Monthly Booking Seasonality",
            labels={"arrival_date_month": "Month", "bookings": "Bookings"},
            template="plotly_white"
        )
        fig_month.update_traces(line_color=COLOR_PRIMARY, line_width=3)
        st.plotly_chart(fig_month, use_container_width=True)

    with col2:
        market_seg = (
            filtered["market_segment"]
            .value_counts()
            .reset_index()
        )
        market_seg.columns = ["market_segment", "bookings"]
        fig_market = px.bar(
            market_seg,
            x="bookings",
            y="market_segment",
            orientation="h",
            title="Bookings by Market Segment",
            color="bookings",
            color_continuous_scale="Blues",
            template="plotly_white"
        )
        st.plotly_chart(fig_market, use_container_width=True)

with tab2:
    st.subheader("Cancellation & Risk Analysis")
    col1, col2 = st.columns(2)

    with col1:
        hotel_cancel = (
            filtered.groupby("hotel")["is_canceled"]
            .mean()
            .mul(100)
            .reset_index(name="cancellation_rate")
        )
        fig_hotel_cancel = px.bar(
            hotel_cancel,
            x="hotel",
            y="cancellation_rate",
            title="Cancellation Rate by Property Type (%)",
            color="hotel",
            color_discrete_sequence=[COLOR_PRIMARY, COLOR_SECONDARY],
            template="plotly_white"
        )
        st.plotly_chart(fig_hotel_cancel, use_container_width=True)

    with col2:
        fig_lead = px.histogram(
            filtered,
            x="lead_time",
            color="is_canceled",
            nbins=30,
            barmode="overlay",
            title="Lead Time vs Cancellation Impact",
            labels={"lead_time": "Lead Time (Days)", "is_canceled": "Cancelled"},
            color_discrete_map={0: COLOR_PRIMARY, 1: COLOR_SECONDARY},
            template="plotly_white"
        )
        st.plotly_chart(fig_lead, use_container_width=True)

with tab3:
    st.subheader("Global Guest Demographics")
    
    # World Map Visualization
    country_df = filtered["country"].value_counts().reset_index()
    country_df.columns = ["country", "bookings"]
    
    fig_map = px.choropleth(
        country_df,
        locations="country",
        color="bookings",
        hover_name="country",
        color_continuous_scale="Viridis",
        title="Global Booking Density (Choropleth Map)",
        template="plotly_white"
    )
    st.plotly_chart(fig_map, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        cust_df = filtered["customer_type"].value_counts().reset_index()
        cust_df.columns = ["customer_type", "bookings"]
        fig_cust = px.pie(
            cust_df,
            names="customer_type",
            values="bookings",
            hole=0.4,
            title="Customer Type Split",
            template="plotly_white"
        )
        st.plotly_chart(fig_cust, use_container_width=True)

    with col2:
        top_10 = country_df.head(10).sort_values(by="bookings")
        fig_top_c = px.bar(
            top_10,
            x="bookings",
            y="country",
            orientation="h",
            title="Top 10 Guest Origin Countries",
            color_continuous_scale="Teal",
            template="plotly_white"
        )
        st.plotly_chart(fig_top_c, use_container_width=True)

with tab4:
    st.subheader("Revenue & Pricing Strategy")
    col1, col2 = st.columns(2)

    with col1:
        adr_month = (
            filtered.groupby("arrival_date_month", observed=False)["adr"]
            .mean()
            .reset_index()
        )
        fig_adr = px.line(
            adr_month,
            x="arrival_date_month",
            y="adr",
            markers=True,
            title="Average Daily Rate (ADR) Trend ($)",
            template="plotly_white"
        )
        fig_adr.update_traces(line_color="#27ae60", line_width=3)
        st.plotly_chart(fig_adr, use_container_width=True)

    with col2:
        fig_stay = px.histogram(
            filtered[filtered["total_nights"] <= 14],
            x="total_nights",
            color="hotel",
            barmode="group",
            title="Length of Stay Distribution (Up to 14 Nights)",
            labels={"total_nights": "Total Nights Stayed"},
            template="plotly_white"
        )
        st.plotly_chart(fig_stay, use_container_width=True)

with tab5:
    st.subheader("Raw Data Explorer")
    
    display_cols = [
        c for c in [
            "hotel", "is_canceled", "lead_time", "arrival_date_year",
            "arrival_date_month", "stays_in_weekend_nights", "stays_in_week_nights",
            "adults", "children", "country", "market_segment", "customer_type", 
            "adr", "estimated_revenue", "total_nights"
        ] if c in filtered.columns
    ]

    st.dataframe(
        filtered[display_cols],
        use_container_width=True,
        height=450
    )

    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Dataset (CSV)",
        data=csv_data,
        file_name="Executive_Hotel_Data.csv",
        mime="text/csv",
    )

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:12px;'>"
    "Enterprise Hotel Revenue Analytics System • Built with Streamlit & Plotly"
    "</div>",
    unsafe_allow_html=True
)
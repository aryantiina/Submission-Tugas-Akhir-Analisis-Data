from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set page configuration
st.set_page_config(layout="wide", page_title="E-Commerce Public Dataset Dashboard")

st.title("Analisis Data E-Commerce Public Dataset")
st.markdown("Dashboard interaktif untuk menjelajahi data pesanan, pengiriman, dan ulasan pelanggan.")

# --- Load Data ---
@st.cache_data
def load_data():
   current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'main_data.csv')
    df = pd.read_csv(file_path)

    # Convert timestamps
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])

    # Extract year and month
    df['purchase_year'] = df['order_purchase_timestamp'].dt.year
    df['purchase_month'] = df['order_purchase_timestamp'].dt.month

    # Calculate delivery time and delay status
    df['delivery_time'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    df['is_delayed'] = df['order_delivered_customer_date'] > df['order_estimated_delivery_date']

    return df

df_original = load_data()
df = df_original.copy()

# --- Sidebar for Filters ---
st.sidebar.header("Filter Data")

# Year filter
all_years = sorted(df['purchase_year'].unique())
selected_years = st.sidebar.multiselect(
    "Pilih Tahun Pembelian",
    options=all_years,
    default=all_years
)
if selected_years:
    df = df[df['purchase_year'].isin(selected_years)]

# Month filter
all_months = sorted(df['purchase_month'].unique())
month_names = {
    1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April', 5: 'Mei', 6: 'Juni',
    7: 'Juli', 8: 'Agustus', 9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
}
selected_months_num = st.sidebar.multiselect(
    "Pilih Bulan Pembelian",
    options=list(month_names.keys()),
    format_func=lambda x: month_names[x],
    default=list(month_names.keys())
)
if selected_months_num:
    df = df[df['purchase_month'].isin(selected_months_num)]

# Review score slider
# Ensure min_score and max_score are integers for the slider
min_score, max_score = int(df['review_score'].min()), int(df['review_score'].max())
selected_score_range = st.sidebar.slider(
    "Rentang Rating Pelanggan",
    min_value=min_score,
    max_value=max_score,
    value=(min_score, max_score)
)
df = df[(df['review_score'] >= selected_score_range[0]) & (df['review_score'] <= selected_score_range[1])]

# Delivery delay filter
delivery_status_options = ['Semua', 'Tertunda', 'Tidak Tertunda']
selected_delivery_status = st.sidebar.selectbox(
    "Status Keterlambatan Pengiriman",
    options=delivery_status_options,
    index=0
)
if selected_delivery_status == 'Tertunda':
    df = df[df['is_delayed'] == True]
elif selected_delivery_status == 'Tidak Tertunda':
    df = df[df['is_delayed'] == False]

# Payment type filter
all_payment_types = df['payment_type'].dropna().unique()
selected_payment_types = st.sidebar.multiselect(
    "Pilih Jenis Pembayaran",
    options=all_payment_types,
    default=all_payment_types
)
if selected_payment_types:
    df = df[df['payment_type'].isin(selected_payment_types)]


st.subheader("Data Hasil Filter")
st.dataframe(df.head(10)) # Display first 10 rows of filtered data


# --- Visualizations ---

st.header("Visualisasi Data")

# Plot 1: Persentase Keterlambatan Pengiriman Berdasarkan Bulan
st.subheader("Persentase Keterlambatan Pengiriman Berdasarkan Bulan")
if not df.empty and 'purchase_month' in df.columns:
    monthly_delay = df.groupby('purchase_month')['is_delayed'].agg(total_orders='count', delayed_orders='sum').reset_index()
    monthly_delay['delay_percentage'] = (monthly_delay['delayed_orders'] / monthly_delay['total_orders']) * 100

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(x='purchase_month', y='delay_percentage', data=monthly_delay, palette='viridis', ax=ax1)
    ax1.set_title('Persentase Keterlambatan Pengiriman Berdasarkan Bulan Pembelian')
    ax1.set_xlabel('Bulan Pembelian')
    ax1.set_ylabel('Persentase Keterlambatan (%)')
    ax1.set_xticks(ticks=range(1, 13))
    ax1.set_xticklabels([month_names.get(i, '') for i in range(1, 13)], rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig1)
else:
    st.warning("Tidak ada data atau kolom 'purchase_month' untuk ditampilkan setelah filter.")

# Plot 2: Hubungan Waktu Pengiriman dan Rating Pelanggan
st.subheader("Hubungan Waktu Pengiriman dan Rating Pelanggan")
if not df.empty and 'delivery_time' in df.columns and 'review_score' in df.columns:
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    sns.regplot(x='delivery_time', y='review_score', data=df, scatter_kws={'alpha':0.3}, line_kws={'color':'red'}, ax=ax2)
    ax2.set_title('Hubungan Waktu Pengiriman dan Rating Pelanggan')
    ax2.set_xlabel('Waktu Pengiriman (Hari)')
    ax2.set_ylabel('Rating Pelanggan')
    st.pyplot(fig2)
else:
    st.warning("Tidak ada data atau kolom 'delivery_time'/'review_score' untuk menampilkan hubungan waktu pengiriman dan rating pelanggan.")

# Plot 3: Distribusi Rating Pelanggan
st.subheader("Distribusi Rating Pelanggan")
if not df.empty and 'review_score' in df.columns:
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.histplot(df['review_score'], bins=5, kde=True, ax=ax3)
    ax3.set_title('Distribusi Rating Pelanggan')
    ax3.set_xlabel('Rating Pelanggan')
    ax3.set_ylabel('Jumlah')
    st.pyplot(fig3)
else:
    st.warning("Tidak ada data atau kolom 'review_score' untuk menampilkan distribusi rating pelanggan.")

# Plot 4: Top 10 Payment Types
st.subheader("Top 10 Jenis Pembayaran")
if not df.empty and 'payment_type' in df.columns:
    top_payment_types = df['payment_type'].value_counts().head(10)
    if not top_payment_types.empty:
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        sns.barplot(x=top_payment_types.index, y=top_payment_types.values, palette='coolwarm', ax=ax4)
        ax4.set_title('Top 10 Jenis Pembayaran')
        ax4.set_xlabel('Jenis Pembayaran')
        ax4.set_ylabel('Jumlah Transaksi')
        ax4.tick_params(axis='x', rotation=45)
        st.pyplot(fig4)
    else:
        st.warning("Tidak ada jenis pembayaran untuk ditampilkan setelah filter.")
else:
    st.warning("Tidak ada data atau kolom 'payment_type' untuk menampilkan jenis pembayaran.")

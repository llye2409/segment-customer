import streamlit as st
import joblib
import json
import pandas as pd
import plotly.express as px


# ------------------------------
# Load trained model, scaler, and bins (using cache)
# ------------------------------
@st.cache_resource
def load_model_and_bins():
    try:
        model = joblib.load("models/kmeans_rfm_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        bins = joblib.load("models/rfm_bins.pkl")
        return model, scaler, bins
    except Exception as e:
        st.error(f"Lỗi khi tải model/scaler: {e}")
        st.stop()

model, scaler, bins = load_model_and_bins()
r_bins, f_bins, m_bins = bins["r_bins"], bins["f_bins"], bins["m_bins"]


# ------------------------------
# Read JSON file containing cluster label information
# ------------------------------
@st.cache_data
def load_cluster_labels():
    try:
        with open("data/cluster_labels.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Lỗi đọc file cluster_labels.json: {e}")
        st.stop()

cluster_labels = load_cluster_labels()


def inject_custom_css():
    st.markdown("""
        <style>
        .main {
            background-color: #ffffff;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stSelectbox > div {
            border-radius: 8px;
        }
        .stTextInput > div > div {
            border-radius: 8px;
        }
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
        }
        </style>
    """, unsafe_allow_html=True)

# ------------------------------
# Customer Segmentation Class: Data processing and clustering from transaction data
# ------------------------------
class CustomerSegmentation:
    def __init__(self):
        self.model = model
        self.scaler = scaler
        self.r_bins = r_bins
        self.f_bins = f_bins
        self.m_bins = m_bins

    def clean_data(self, df):
        try:
            df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce').fillna(0).astype(int)
            df['TotalSales'] = pd.to_numeric(df['TotalSales'], errors='coerce').fillna(0).astype(float)
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
            df = df.dropna(subset=['CustomerID', 'InvoiceDate', 'TotalSales'])
            df = df[df['TotalSales'] > 0]
            df = df.drop_duplicates()
            return df
        except Exception as e:
            st.error(f"Lỗi xử lý dữ liệu: {e}")
            return pd.DataFrame()

    def calculate_rfm(self, df):
        try:
            max_date = df['InvoiceDate'].max().date()
            rfm_df = df.groupby('CustomerID').agg(
                Recency=('InvoiceDate', lambda x: (max_date - x.max().date()).days),
                Frequency=('CustomerID', 'count'),
                Monetary=('TotalSales', 'sum')
            ).reset_index()
            rfm_df['Monetary'] = rfm_df['Monetary'].fillna(0)
            return rfm_df
        except Exception as e:
            st.error(f"Lỗi tính toán RFM: {e}")
            return pd.DataFrame()

    def prepare_rfm_features(self, rfm_df, r_bins, f_bins, m_bins):
        try:
            df_copy = rfm_df.copy()
            df_copy['Recency'] = df_copy['Recency'].clip(lower=r_bins[0], upper=r_bins[-1])
            df_copy['Frequency'] = df_copy['Frequency'].clip(lower=f_bins[0], upper=f_bins[-1])
            df_copy['Monetary'] = df_copy['Monetary'].clip(lower=m_bins[0], upper=m_bins[-1])
            
            df_copy['R'] = pd.cut(df_copy['Recency'], bins=r_bins, 
                                  labels=[4, 3, 2, 1], include_lowest=True).astype(int)
            df_copy['F'] = pd.cut(df_copy['Frequency'], bins=f_bins, 
                                  labels=[1, 2, 3, 4], include_lowest=True).astype(int)
            df_copy['M'] = pd.cut(df_copy['Monetary'], bins=m_bins, 
                                  labels=[1, 2, 3, 4], include_lowest=True).astype(int)
    
            X_RFM = df_copy[['R', 'F', 'M']]
            return X_RFM
        except Exception as e:
            st.error(f"Lỗi chuẩn bị dữ liệu RFM: {e}")
            return pd.DataFrame()

    def segment_customers(self, df):
        try:
            df_clean = self.clean_data(df)
            if df_clean.empty:
                return pd.DataFrame()
            rfm_df = self.calculate_rfm(df_clean)
            if rfm_df.empty:
                return pd.DataFrame()
            X_RFM = self.prepare_rfm_features(rfm_df, self.r_bins, self.f_bins, self.m_bins)
            # Scale RFM data after qcut
            X_RFM_scaled = self.scaler.transform(X_RFM)
            # Dự đoán cluster
            rfm_df['Cluster'] = self.model.predict(X_RFM_scaled)
            # Labeling by cluster map
            cluster_map = {0: 'Loyal', 1: 'Regular', 2: 'At-risk', 3: 'No-Potential'}
            rfm_df['Segment'] = rfm_df['Cluster'].map(cluster_map)
            return rfm_df
        except Exception as e:
            st.error(f"Lỗi phân cụm: {e}")
            return pd.DataFrame()

    def summarize_clusters(self, rfm_df):
        try:
            summary = rfm_df.groupby('Cluster').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean',
                'CustomerID': 'count'
            }).rename(columns={'CustomerID': 'Số khách hàng'}).reset_index()
            return summary
        except Exception as e:
            st.error(f"Lỗi tổng hợp nhóm: {e}")
            return pd.DataFrame()

# ------------------------------
# Segmentation Visualize Class: Handling the display of results
# ------------------------------

class SegmentationVisualizer:
    @staticmethod
    def show_summary_info(rfm_df):
        """
        Display the summary information of customer segmentation:
        - Number of unique customers and clusters
        - Average values of Recency, Frequency, Monetary
        - Pie and bar charts showing customer distribution by cluster

        Args:
            rfm_df (pd.DataFrame): DataFrame containing RFM metrics and cluster labels.
        """
        try:
            st.markdown("### 📌 Cluster Overview")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                render_card("Customers", f"{rfm_df['CustomerID'].nunique()}", "👤")
            with col2:
                render_card("Cluster", f"{rfm_df['Cluster'].nunique()}", "📦")
            with col3:
                render_card("Recency.avg", f"{round(rfm_df['Recency'].mean(), 2)}", "⏰")
            with col4:
                render_card("Frequency.avg", f"{round(rfm_df['Frequency'].mean(), 2)}", "🔁")
            with col5:
                render_card("Monetary.avg", f"{round(rfm_df['Monetary'].mean(), 2)}", "💰")

            cluster_counts = rfm_df['Cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Số khách hàng']

            col7, col8 = st.columns(2)
            fig_pie = px.pie(cluster_counts, names='Cluster', values='Số khách hàng', hole=0.4, color='Cluster')
            fig_bar = px.bar(cluster_counts, x='Cluster', y='Số khách hàng', text='Số khách hàng', color='Cluster')
            col7.plotly_chart(fig_pie, use_container_width=True)
            col8.plotly_chart(fig_bar, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying summary: {e}")

    @staticmethod
    def plot_rfm_bar(cluster_summary):
        """
        Create grouped bar chart of RFM metrics per cluster.

        Args:
            cluster_summary (pd.DataFrame): Summary DataFrame with RFM averages per cluster.
        """
        try:
            st.markdown("### 📉 Chart RFM")
            melted = cluster_summary.rename(columns={'Recency': 'R', 'Frequency': 'F', 'Monetary': 'M'}) \
                                     .melt(id_vars=['Cluster'], var_name='Metric', value_name='Value')
            fig = px.bar(melted, x="Cluster", y="Value", color="Metric", barmode="group")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating RFM chart: {e}")

    @staticmethod
    def plot_cluster_scatter(rfm_df):
        """
        Plot scatter chart (Recency vs Frequency), color-coded by Cluster.

        Args:
            rfm_df (pd.DataFrame): DataFrame containing RFM metrics and cluster labels.
        """
        try:
            fig_2d = px.scatter(rfm_df, x="Recency", y="Frequency", color="Cluster",
                                hover_data=["CustomerID", "Monetary"])
            st.plotly_chart(fig_2d, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating scatter plot: {e}")

    @staticmethod
    def show_cluster_table(rfm_df):
        """
        Display an interactive table with filtering by Cluster and CustomerID.

        Args:
            rfm_df (pd.DataFrame): DataFrame containing RFM metrics and cluster labels.
        """
        try:
            st.markdown("### 📋 Detailed Report")
            cluster_options = ['Tất cả'] + sorted(rfm_df['Cluster'].unique().tolist())
            selected_cluster = st.selectbox('Chọn nhóm khách hàng', cluster_options)
            search_id = st.text_input("Tìm theo CustomerID")

            filtered_df = rfm_df.copy()
            if selected_cluster != 'Tất cả':
                filtered_df = filtered_df[filtered_df['Cluster'] == int(selected_cluster)]

            if search_id:
                try:
                    cid = int(search_id)
                    filtered_df = filtered_df[filtered_df['CustomerID'] == cid]
                except:
                    st.warning("⚠️ CustomerID must be a number.")

            with st.expander("📋 View detailed data table"):
                st.dataframe(filtered_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying data table: {e}")

    @staticmethod
    def show_cluster_summary(rfm_df):
        try:
            st.markdown("### 📉 RFM Segments Summary")
            total = len(rfm_df)

            # Calculate average RFM per cluster
            summary = rfm_df.groupby('Cluster').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean',
            }).reset_index()

            # Count customers per cluster
            summary['Count'] = rfm_df.groupby('Cluster').size().values
            summary['(%)'] = round(summary['Count'] / total * 100, 2)

            # Compute revenue per customer
            revenue_per_cluster = rfm_df.copy()
            revenue_per_cluster['Revenue'] = revenue_per_cluster['Frequency'] * revenue_per_cluster['Monetary']

            # Sum revenue per cluster
            revenue_summary = revenue_per_cluster.groupby('Cluster')['Revenue'].sum().reset_index()

            # Merge revenue with summary
            summary = summary.merge(revenue_summary, on='Cluster')

            # Calculate revenue contribution %
            tong_doanh_thu = summary['Revenue'].sum() if summary['Revenue'].sum() != 0 else 1
            summary['% Revenue'] = round(summary['Revenue'] / tong_doanh_thu * 100, 2)

            # Format numbers
            summary['Recency'] = summary['Recency'].round(1)
            summary['Frequency'] = summary['Frequency'].round(1)
            summary['Monetary'] = summary['Monetary'].round(1)
            summary['Revenue'] = summary['Revenue'].map('{:,.0f}'.format)  # comma separator, no decimals

            # Display the summary table
            st.dataframe(summary.rename(columns={'Cluster': 'Nhóm'}), use_container_width=True)

        except Exception as e:
            st.error(f"Error in summary table: {e}")



    @staticmethod
    def suggest_actions(cluster_summary, cluster_labels):
        """
        Provide recommended actions and descriptions for each customer cluster.

        Args:
            cluster_summary (pd.DataFrame): Summary data per cluster.
            cluster_labels (dict): Dictionary with cluster labels, descriptions, and suggestions.
        """
        try:
            st.markdown("### 💡 Suggestions")
            for _, row in cluster_summary.iterrows():
                cluster_id = str(int(row['Cluster']))
                label = cluster_labels.get(cluster_id)
                group_name = label.get('name', f"Cluster {cluster_id}") if label else f"Cluster {cluster_id}"
                mo_ta = label.get('desc', 'No description') if label else 'No description'
                goi_y = label.get('traits', 'No suggestions') if label else 'No suggestions'

                with st.expander(group_name):
                    st.markdown(f"**Description:** {mo_ta}")
                    st.markdown(f"**Suggestions:** {goi_y}")
            st.markdown("---")
        except Exception as e:
            st.error(f"Error showing suggestions: {e}")

def render_card(title, value, icon):
    """
    Render a styled metric card with an icon, title, and value.

    Args:
        title (str): Title of the card.
        value (str): Value to display.
        icon (str): Emoji or icon representing the metric.
    """
    st.markdown(
        f"""
        <div style="background-color: #f9f9f9; padding: 1rem 1.5rem; border-radius: 15px;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.05); text-align: center;">
            <div style="font-size: 2rem;">{icon}</div>
            <div style="font-size: 1rem; font-weight: 600; color: #555;">{title}</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #111;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def prepare_rfm_features(df):
    df = df.copy()
    df['Recency'] = df['Recency'].clip(lower=r_bins[0], upper=r_bins[-1])
    df['Frequency'] = df['Frequency'].clip(lower=f_bins[0], upper=f_bins[-1])
    df['Monetary'] = df['Monetary'].clip(lower=m_bins[0], upper=m_bins[-1])
    
    df['R'] = pd.cut(df['Recency'], bins=r_bins, labels=[4, 3, 2, 1], include_lowest=True).astype(int)
    df['F'] = pd.cut(df['Frequency'], bins=f_bins, labels=[1, 2, 3, 4], include_lowest=True).astype(int)
    df['M'] = pd.cut(df['Monetary'], bins=m_bins, labels=[1, 2, 3, 4], include_lowest=True).astype(int)
    
    return df[['R', 'F', 'M']]
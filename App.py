import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

st.set_page_config(page_title="🎯 RFM Clustering App", layout="centered", page_icon="📊")

from lib.mylib import *
# Apply css
inject_custom_css()

st.markdown("<h2 style='text-align:center; color:#4B8BBE;'>📊 RFM Customer Segmentation</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Phân loại khách hàng dựa trên chỉ số Recency, Frequency, và Monetary.</p>", unsafe_allow_html=True)
st.markdown("---")

# Allows users to choose between manual input and file upload
mode = st.radio("Chọn chế độ nhập", ("Nhập thủ công RFM", "Tải file dữ liệu giao dịch"))

if mode == "Nhập thủ công RFM":
    with st.form("rfm_input_form"):
        st.subheader("📥 Nhập chỉ số RFM của khách hàng")
        
        with st.expander("📘 Giải thích các chỉ số", expanded=False):
            st.markdown("""
            - **Recency (🕒)**: Số ngày kể từ lần mua gần nhất. Càng thấp càng tốt.
            - **Frequency (🔁)**: Số lần mua hàng. Càng cao càng tốt.
            - **Monetary (💰)**: Tổng số tiền đã chi tiêu. Càng cao càng tốt.
            """)

        col1, col2, col3 = st.columns(3)
        recency = col1.number_input("🕒 Recency (ngày)", min_value=0, step=1)
        frequency = col2.number_input("🔁 Frequency (lần)", min_value=1, step=1)
        monetary = col3.number_input("💰 Monetary (VNĐ)", min_value=0.0, format="%.2f")

        submitted = st.form_submit_button("🚀 Dự đoán phân cụm")

    # ================== Processing and Display ==================
    if submitted:
        with st.spinner("🔍 Đang phân tích dữ liệu..."):
            new_customer = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
            X_new = prepare_rfm_features(new_customer)
            X_scaled = scaler.transform(X_new)
            cluster = model.predict(X_scaled)[0]

            # ==== Tính xác suất gần đúng dựa trên khoảng cách đến tâm cụm ====
            distances = np.linalg.norm(model.cluster_centers_ - X_scaled, axis=1)
            inv_distances = 1 / (distances + 1e-6)  # tránh chia cho 0
            probabilities = inv_distances / inv_distances.sum()

            

            segment_info = cluster_labels[str(cluster)]
            segment_name = segment_info["name"]
            segment_desc = segment_info["desc"]
            segment_traits = segment_info["traits"]

            result_df = pd.concat([new_customer, X_new], axis=1)
            result_df['Cluster'] = cluster
            result_df['Segment'] = segment_name

            # Kết quả
            st.toast(f"📌 Khách hàng thuộc nhóm: {segment_name}", icon="🎯")
            st.success(f"**📌 Group: {segment_name}**")
            st.markdown(f"**🧬 Describe:**\n{segment_desc}")
            st.markdown(f"**💡 Suggest:** {segment_traits}")

            # ======== Xác suất dự đoán ========
            with st.expander("📊 Xem xác suất dự đoán (nâng cao)", expanded=False):
                proba_df = pd.DataFrame({
                    "Cluster": [cluster_labels[str(i)]["name"] for i in range(len(probabilities))],
                    "Probability (%)": np.round(probabilities * 100, 2)
                })


            # Chi tiết & Tải xuống
            with st.expander("📄 Xem dữ liệu chi tiết"):
                st.dataframe(result_df)

            csv_buffer = StringIO()
            result_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Tải kết quả (.CSV)",
                data=csv_buffer.getvalue(),
                file_name="rfm_result.csv",
                mime="text/csv",
                use_container_width=True
            )

else:
    st.markdown("### 📁 Tải lên file dữ liệu giao dịch")
    expected_columns = {'CustomerID', 'InvoiceDate', 'TotalSales'}
    uploaded_file = st.file_uploader("📁 Tải lên file dữ liệu (.csv)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if not expected_columns.issubset(df.columns):
                st.warning(f"⚠️ File phải chứa các cột: {expected_columns}")
            else:
                st.success("✅ Dữ liệu đã được tải thành công!")
                segmentor = CustomerSegmentation()
                visualizer = SegmentationVisualizer()
                rfm_df = segmentor.segment_customers(df)
                if rfm_df is not None and not rfm_df.empty:
                    visualizer.show_summary_info(rfm_df)
                    cluster_summary = segmentor.summarize_clusters(rfm_df)
                    if not cluster_summary.empty:
                        visualizer.plot_rfm_bar(cluster_summary)
                        visualizer.plot_cluster_scatter(rfm_df)
                        visualizer.show_cluster_summary(rfm_df)
                        visualizer.suggest_actions(cluster_summary, cluster_labels)
                    
                    visualizer.show_cluster_table(rfm_df)
        except Exception as e:
            st.error(f"❌ Đã xảy ra lỗi khi đọc file: {e}")
    else:
        st.info("📄 Hãy tải lên file dữ liệu để bắt đầu phân tích. File cần bao gồm các cột: CustomerID, InvoiceDate, TotalSales.")
        st.markdown("""
        ### 📝 Hướng dẫn chuẩn bị dữ liệu:
        - **CustomerID**: Mã định danh duy nhất của khách hàng (số nguyên).
        - **InvoiceDate**: Ngày giao dịch (định dạng ngày/tháng/năm).
        - **TotalSales**: Tổng giá trị hóa đơn (số thực).

        👉 Dữ liệu cần đảm bảo:
        - Không có giá trị thiếu hoặc sai định dạng.
        - Không chứa giao dịch có giá trị âm hoặc bằng 0.

        ### 📦 Ví dụ mẫu:
        | CustomerID | InvoiceDate | TotalSales |
        |------------|-------------|------------|
        | 12345      | 2023-10-01  | 250.00     |
        | 67890      | 2023-09-15  | 120.50     |

        Bạn có thể tải về file mẫu để tham khảo: [📥 Tải file mẫu](https://drive.google.com/uc?export=download&id=1dd5QbGEWF1ONFolAXDH6KFDJsOjm5zz1)
        """)
    
    st.markdown("---")
    st.markdown("Developed with ❤️ by [anhkuan / Team Duy-An-Gulist] | Powered by RFM Analysis & KMeans")

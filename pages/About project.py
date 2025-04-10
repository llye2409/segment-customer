import streamlit as st
import pandas as pd
from PIL import Image

# --- Cấu hình giao diện chính ---
st.set_page_config(page_title="🎯 Project Overview", layout="centered", page_icon="📊")

st.title("📊 Customers Segmentation Project - Cửa Hàng X")

# --- 1. VẤN ĐỀ KINH DOANH ---
st.header("1. Project Objective")
st.markdown("""
Mục tiêu chính của dự án là giúp **Cửa hàng X tối ưu hóa doanh thu** thông qua các hành động sau:

- Bán được nhiều hàng hóa hơn.
- Tiếp cận đúng đối tượng khách hàng phù hợp với từng sản phẩm.
- Chăm sóc và làm hài lòng khách hàng để họ quay lại mua sắm nhiều lần.

Dự án tập trung vào phân tích hành vi mua sắm và **phân khúc khách hàng dựa trên dữ liệu giao dịch thực tế**, từ đó đề xuất chiến lược phù hợp cho từng nhóm khách hàng nhằm nâng cao trải nghiệm và xây dựng lòng trung thành.
""")

# --- 2. TỔNG QUAN DỮ LIỆU ---
st.header("2. Data")

st.markdown("""
Dữ liệu gồm:
- **Thông tin sản phẩm:** productId, productName, price, Category
- **Thông tin giao dịch:** Member_number, Date, items, Total_Revenue
- **Thông tin thời gian:** 01/01/2014 - 30/12/2015
""")

st.subheader("📌 Data sample")
st.image('images/data-sample.png', caption="Dữ liệu")

# --- 3. TIỀN XỬ LÝ DỮ LIỆU ---
st.header("3. Data preprocessing")
st.markdown("""
Các bước xử lý gồm:
- Hợp nhất và làm sạch dữ liệu (xử lý null, loại trùng lặp)
- Chuyển đổi `Date` sang dạng thời gian
- Trích xuất `month` và `weekday`
- Tính toán chỉ số **RFM** cho mỗi khách hàng:
  - **Recency:** Ngày gần nhất mua hàng
  - **Frequency:** Số lần mua hàng
  - **Monetary:** Tổng tiền đã chi
- Chuẩn hóa RFM
- Xử lý outliers
""")

# --- 4. TRỰC QUAN HÓA DỮ LIỆU ---
st.header("4. Data visualization")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Phân bố các biến")
    st.image("images/price_distribution.jpg", caption="Phân bố giá sản phẩm")
    st.image("images/total_revenue_distribution.jpg", caption="Phân bố tổng doanh thu")

with col2:
    st.subheader("📊 Biểu đồ tần suất & xu hướng")
    st.image("images/product_category_bar.jpg", caption="Giao dịch theo danh mục sản phẩm")
    st.image("images/monthly_sales_trend.jpg", caption="Doanh số theo tháng")
    st.image("images/weekday_sales_trend.jpg", caption="Doanh số theo ngày trong tuần")

st.subheader("🔍 Mối quan hệ giữa các biến số")
st.image("images/scatter_revenue_vs_member.jpg", caption="Tổng doanh thu vs số thành viên")
st.image("images/scatter_rfm_2d.jpg", caption="Recency vs Frequency")

# --- 5. PHÂN CỤM KHÁCH HÀNG ---
st.header("5. Modeling")

st.markdown("""
- Mỗi khách hàng được gán điểm RFM từ 1 đến 4 dựa trên tứ phân vị.
- Chuẩn hóa dữ liệu RFM trước khi phân cụm.
- Sử dụng phương pháp Elbow để tìm số cụm K tối ưu.
""")

st.subheader("📊 Trực quan cụm khách hàng")

st.image("images/customer_segments_treemap.jpg", caption="Treemap: Kích thước từng cụm KMeans")
st.image("images/rfm_cluster_distributions.png", caption="rfm cluster distributions")
st.image("images/rfm_cluster_trend_by_month.png", caption="rfm cluster trend by_month")

# --- Kết thúc ---
st.markdown("---")
st.info("📍 Dự án này được phát triển với mục tiêu tăng doanh thu, giữ chân khách hàng và tối ưu chiến lược tiếp thị của Cửa hàng X.")


st.markdown("Developed with ❤️ by [anhkuan / Team Duy-An-Gulist] | Powered by RFM Analysis & KMeans")

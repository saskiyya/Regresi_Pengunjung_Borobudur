import pandas as pd
import joblib
import streamlit as st

st.set_page_config(
	page_title="Prediksi Pengunjung Borobudur"
)

model_forest = joblib.load("model_forest.joblib")

st.title("Prediksi Penjunjung Borobudur")
st.markdown("Aplikasi Machine Learning untuk Memprediksi Pengunjung Borobudur")

hari_type = st.pills("Hari Type", ["weekend", "weekday"], default="weekend")
musim = st.pills("Musim", ["kemarau", "hujan"], default="hujan")
suhu_rata_rata = st.slider("Suhu Rata Rata", 20.0, 35.0, 25.0)
ada_event_budaya = st.pills("Ada Event Budaya", ["ya", "tidak"], default="ya")
harga_tiket_ribu = st.slider("Harga Tiket", 50.0, 100.0, 60.0)

if st.button("Prediksi"):
	data_baru = pd.DataFrame([[hari_type, musim, suhu_rata_rata, ada_event_budaya,harga_tiket_ribu]], 
                         columns=["hari_type", "musim",	"suhu_rata_rata", "ada_event_budaya","harga_tiket_ribu"])
	prediksi = model_forest.predict(data_baru)[0]
	st.success(f"Model memprediksi jumlah pengunjung borobudur adalah {prediksi:.0f}")
	st.balloons()

st.divider()
st.caption("Dibuat Oleh Saskia Humaira")


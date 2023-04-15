import pandas as pd
import streamlit as st
import plotly.express as px
from matplotlib import category
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# membuka file csv
data = pd.read_csv('data/trending_new.csv')

# Memilih kolom views sebagai target prediksi dan beberapa kolom lain sebagai fitur
X = data[['like', 'dislike', 'comment']]
y = data['view']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model regresi linear dan melatih model
model = LinearRegression()
model.fit(X_train, y_train)

# Melakukan prediksi pada data uji
y_pred = model.predict(X_test)

# Menampilkan hasil prediksi pada antarmuka
st.title('Prediksi Jumlah Views')
st.write('Masukan data fitur untuk melakukan prediksi')
likes = st.number_input("Likes", 0)
dislikes = st.number_input("Dislikes", 0)
comments = st.number_input("Comments", 0)

if st.button("Prediksi"):
    pred = model.predict([[likes, dislikes, comments]])
    st.success(f"Prediksi jumlah views = {int(pred[0])}")


px.defaults.template = 'plotly_dark'
px.defaults.color_continuous_scale = 'reds'

img = Image.open('assets/yutup.png')
st.sidebar.image(img)

# input tanggal
min_date = pd.to_datetime(data['trending_date']).min()
max_date = pd.to_datetime(data['trending_date']).max()
start_date, end_date = st.sidebar.date_input(label='Rentang Waktu',
                                            min_value=min_date.date(),
                                            max_value=max_date.date(),
                                            value=[min_date.date(), max_date.date()])

# input kategori
categories = ["All Categories"] + list(data['category'].value_counts().keys().sort_values())
category = st.sidebar.selectbox(label='Kategori', options=categories)

# filters
outputs = data[(pd.to_datetime(data['trending_date']) >= pd.to_datetime(start_date)) &
                (pd.to_datetime(data['trending_date']) <= pd.to_datetime(end_date))]
if category != "All Categories":
    outputs = outputs[outputs['category'] == category]

# visualisasi dengan bar chart
st.header(':video_camera: Channel')
bar_data = outputs['channel_name'].value_counts().nlargest(10)
fig = px.bar(bar_data, color=bar_data, orientation='h', title=f'Channel Terpopuler dari Kategori {category}')
st.plotly_chart(fig)

# visualisasi dengan scatter plot
st.header(':bulb: Engagement')
col1, col2 = st.columns(2)
metrics_coice1 = ['like', 'comment']
metrics_coice2 = ['dislike', 'comment']
choice1 = col1.selectbox('Horizontal', options=metrics_coice1)
choice2 = col2.selectbox('Vertical', options=metrics_coice2)
fig = px.scatter(outputs,
                x=choice1,
                y=choice2,
                size='view',
                hover_name='channel_name',
                hover_data=['title'],
                title=f'Engagement of {choice1.title()} and {choice2.title()}')
st.plotly_chart(fig)

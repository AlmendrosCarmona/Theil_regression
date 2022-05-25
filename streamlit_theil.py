import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression, TheilSenRegressor
import plotly.express as px
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import plotly.figure_factory as ff


st.set_page_config(layout="wide")

st.title("Theil-Senn Analisis")
st.text("Upload any number of files: ")
uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)

for uploaded_file in uploaded_files:
    df = pd.read_csv(uploaded_file, delimiter=';')
    df = df[["TempVAT", "In"]]
    df.dropna(subset=["In"], inplace=True)
    # df =df[df["TempVAT"]>=48]
    col1, col2, col3 = st.columns([20, 5, 5])
    slider = col1.slider("Seleccionar rango de datos: ", int(df["TempVAT"].min()), int(
        df["TempVAT"].max()+1), (int(df["TempVAT"].min()), int(df["TempVAT"].max()+1)))
    df = df[df["TempVAT"] >= slider[0]]
    df = df[df["TempVAT"] <= slider[1]]
    fig = px.scatter(df, x='TempVAT', y='In', marginal_x="histogram", marginal_y='histogram')
    fig.update_yaxes(range=[-5, 60])
    # Histograma In
    group_labels = ['In']
    #fig2 = px.histogram(df, x='In', marginal='box')

    regressor = LinearRegression()
    X = np.array(df['TempVAT']).reshape(-1, 1)
    y = np.array(df['In']).reshape(-1, 1)
    regressor.fit(X, y)
    regressor_theil = TheilSenRegressor()
    regressor_theil.fit(X, df['In'])
    lineal_y = regressor.predict(X)
    theil_y = regressor_theil.predict(X)
    lineal_y = lineal_y.flatten().transpose()
    theil_y = theil_y.flatten().transpose()
    df_lineal = pd.DataFrame({'x': df["TempVAT"], 'y': lineal_y})
    df_theil = pd.DataFrame({'x': df["TempVAT"], 'y': theil_y})
    fig.add_trace(px.line(df_lineal, x='x', y='y', color_discrete_sequence=['red']).data[0])
    fig.add_trace(px.line(df_theil, x='x', y='y', color_discrete_sequence=['orange']).data[0])
    col1, col2, col3 = st.columns([20, 5, 5])
    col1.header(uploaded_file.name)
    col1.plotly_chart(fig)
    col2.header("Features")
    col2.header("")
    col3.header("")
    col3.header("")
    col3.header("")
    col3.text("")
    col2.metric(label='Slope Lineal', value=round(regressor.coef_[0, 0], 4))
    col3.metric(label='Intercept Lineal', value=round(regressor.intercept_[0], 4))
    col2.metric(label='Slope Theil', value=round(regressor_theil.coef_[0], 4))
    col3.metric(label='Intercept Theil', value=round(regressor_theil.intercept_, 4))

    m_l = round(regressor.coef_[0, 0], 4)
    m_t = round(regressor_theil.coef_[0], 4)

    #Angle = round(np.arccos((1+m_l*m_t)/((1+m_t*m_t)**(1/2)*(1+m_l*m_l)**(1/2)))*360/np.pi,3)
    Angle = round(np.arccos((1+m_l*m_t)/((1+m_t*m_t)**(1/2)*(1+m_l*m_l)**(1/2)))*360/np.pi, 3)
    col2.metric(label='Angle (deg)', value=Angle)

    # Calculo de distancia
    df_theil["Distance"] = df_lineal["y"] - df_theil["y"]
    col3.metric(label="Distance", value=round(df_theil["Distance"].mean(), 3))
    dist = col2.number_input("Dist In", key=uploaded_file.name)
    button = col3.selectbox("Histogram In: ", ('None', 'Histogram'), key=uploaded_file.name)
    if button == 'Histogram':
        fig3 = ff.create_distplot([df['In']], group_labels, bin_size=.5,
                                  curve_type='kde')
        col1.plotly_chart(fig3)

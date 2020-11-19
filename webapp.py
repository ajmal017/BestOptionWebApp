#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:34:22 2020

@author: Marco
"""

import yfinance as yf
import streamlit as st
from pandas_datareader import data as pdr
yf.pdr_override() 
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats  as si
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import datetime

def analytical_solution(S01,S02,sigma1,sigma2,q1,q2,r,T,rho):
    sigma_hat = np.sqrt (sigma1**2+sigma2**2-2*rho*sigma1*sigma2)
    d1 = (np.log(S01/S02)+(q2-q1+sigma_hat**2/2)*T)/(sigma_hat*np.sqrt(T))
    d2 = d1 - sigma_hat*np.sqrt(T)
    analytical = S01*np.exp(-q1*T)*si.norm.cdf(d1, 0.0, 1.0)-S02*np.exp(-q2*T)*si.norm.cdf(d2, 0.0, 1.0)+S02*np.exp(-q2*T)
    return analytical


def final_value(S01,S02,sigma1,sigma2,q1,q2,r,T,rho):
    WT1 = rd.gauss(0,1.0)
    WT2 = rd.gauss(0,1.0)
    ST1 = S01*np.exp((r-q1-0.5*sigma1**2)*T+sigma1*np.sqrt(T)*WT1)
    ST2 = S02*np.exp((r-q2-0.5*sigma2**2)*T+sigma2*rho*np.sqrt(T)*WT1+sigma2*np.sqrt(1-rho**2)*np.sqrt(T)*WT2)
    return ST1, ST2

def monte_carlo(num_sim,S01,S02,sigma1,sigma2,q1,q2,r,T,rho):
    nST1 = [];
    nST2 = [];
    nPayoff = []
    for i in range (num_sim):
        ST = final_value(S01,S02,sigma1,sigma2,q1,q2,r,T,rho)
        nST1.append(ST[0])
        nST2.append(ST[1])
        nPayoff.append(np.maximum(ST[0],ST[1]))
    estimate = 1/num_sim*np.exp(-r*T)*np.sum(nPayoff)
    return estimate

st.write("""
# Best Asset Option Price Calculator
Montecarlo estimate and analytical solution for best asset option

""")
st.sidebar.header('User input')
def user_input_side():
    num_digits = int(st.sidebar.slider("Number of decimal digits ",1,10,1))
    S01 = float(st.sidebar.text_input('S01', 10))
    S02 = float(st.sidebar.text_input('S02', 20))
    sigma1 = float(st.sidebar.text_input('Volatility 1', 0.2))
    sigma2 = float(st.sidebar.text_input('Volatility 2', 0.3))
    rho = float(st.sidebar.text_input('Correlation', 0.25))
    q1 = float(st.sidebar.text_input('Dividend Yield 1', 0.02))
    q2 = float(st.sidebar.text_input('Dividend Yield 2', 0.03))
    r = float(st.sidebar.text_input('Risk free rate', 0.04))
    T = float(st.sidebar.text_input('Maturity', 3))
    num_sim = int(st.sidebar.slider('Num sim',100,10000, 100))
    return num_sim,S01,S02,sigma1,sigma2,q1,q2,r,T,rho, num_digits

def user_input_side_real():
    num_digits = int(st.sidebar.slider("Number of decimal digits ",1,10,1))
    stock1=st.sidebar.text_input('S1','AAPL')
    stock2=st.sidebar.text_input('S2','SPY')
    r = float(st.sidebar.text_input('Risk free rate', 0.04))
    num_sim = int(st.sidebar.slider('Num sim',100,10000, 100))
    return stock1, stock2, r, num_sim, num_digits

def real_data(stock1, stock2):
    stock = stock1 + ' ' + stock2
    

    end_time = datetime.datetime.now()
    start_time = datetime.datetime.now() - datetime.timedelta(days=2*365)
    
    data = pdr.get_data_yahoo(stock, start_time, end_time)
    time_series1 = data['Adj Close'][stock1]
    time_series2 = data['Adj Close'][stock2]
    
    st.line_chart(data['Adj Close'])
    data['Adj Close'][stock1].plot()
    plt.xlabel("Date")
    plt.ylabel("Adjusted")
    plt.title(stock1+"Price data")
    plt.show()
    
    data['Adj Close'][stock2].plot()
    plt.xlabel("Date")
    plt.ylabel("Adjusted")
    plt.title(stock2+"Price data")
    plt.show()
    
    #calculation of daily returns
    stock1_dr = data['Adj Close'][stock1].pct_change()
    stock2_dr = data['Adj Close'][stock2].pct_change()
    stock1_ldr = np.log(data['Adj Close'][stock1]/data['Adj Close'][stock1].shift(1))
    stock2_ldr = np.log(data['Adj Close'][stock2]/data['Adj Close'][stock2].shift(1))
    
    stock1_ldr.drop(stock1_ldr.index[0])
    stock2_ldr.drop(stock2_ldr.index[0])
    
    data_returns= pd.DataFrame( {'Stock1 ldr': stock1_ldr, 'Stock2 ldr': stock2_ldr})
    
    #calculation of daily vol and corr
    rho = data_returns.corr()['Stock1 ldr']['Stock2 ldr']
    var_stock1 = np.std(stock1_ldr)
    var_stock2 = np.std(stock2_ldr)
    
    #calculation of annual vol
    sigma1 = np.sqrt(252)*var_stock1
    sigma2 = np.sqrt(252)*var_stock2
    
    #q1, q2 = real_dividends (stock1, stock2, time_series1[-1], time_series2[-1])
    
    return time_series1[-1], time_series2[-1], sigma1, sigma2, rho, 

def error_plot(S01, S02, sigma1, sigma2, rho, r, T, q1, q2):
    error = []
    for num_sim in range (100,10000,100):
        estimate = monte_carlo(num_sim,S01,S02,sigma1,sigma2,q1,q2,r,T,rho)
        analytical = analytical_solution(S01,S02,sigma1,sigma2,q1,q2,r,T,rho)
        dummy = estimate - analytical
        error.append(dummy)
    st.line_chart(error)

yes = st.sidebar.selectbox("Which type of input would you like to use?", ('Manual Input', 'Real Data'))

if yes == 'Manual Input' : 
    num_sim,S01,S02,sigma1,sigma2,q1,q2,r,T,rho, num_digits = user_input_side()
    estimate = round(monte_carlo(num_sim,S01,S02,sigma1,sigma2,q1,q2,r,T,rho), num_digits)
    analytical = round(analytical_solution(S01,S02,sigma1,sigma2,q1,q2,r,T,rho), num_digits)
    
    st.subheader('Montecarlo estimate')
    st.write("The estimate is ", estimate)
    
    st.subheader('Analytical solution')
    st.write("The analytical solution is ", analytical)

    st.subheader('Estimation error')
    st.write("The estimation absolute error", round(np.abs(analytical - estimate),num_digits))
    
    st.subheader('Error convergence')
    
    if st.button('Show convergence plot'):     
        error_plot(S01, S02, sigma1, sigma2, rho, r, T, q1, q2)
else:
    stock1, stock2, r, num_sim, num_digits = user_input_side_real()
    S01, S02, sigma1, sigma2, rho = real_data(stock1, stock2)
    T = 2 #default
    q1 = 0 #default
    q2 = 0 #default
    estimate = round(monte_carlo(num_sim,S01,S02,sigma1,sigma2,q1,q2,r,T,rho), num_digits)
    st.subheader('Montecarlo estimate with real data')
    st.write("The estimate is ", estimate)
    st.subheader('Error convergence')
    if st.button('Show convergence plot'):     
        error_plot(S01, S02, sigma1, sigma2, rho, r, T, q1, q2)
   




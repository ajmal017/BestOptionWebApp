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
#from mpl_toolkits.mplot3d import Axes3D
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
    nST1 = []
    nST2 = []
    nPayoff = []
    for i in range (num_sim):
        ST = final_value(S01,S02,sigma1,sigma2,q1,q2,r,T,rho)
        nST1.append(ST[0])
        nST2.append(ST[1])
        nPayoff.append(np.exp(-r*T)*np.maximum(ST[0],ST[1]))
    estimate = 1/num_sim*np.sum(nPayoff)
    confidence = pd.DataFrame(np.zeros((2,3)), columns = ['Lower Band','Estimate','Upper Band'], index = ['95%','99%'])
    confidence.iloc[0,0] = estimate-1.96*np.std(nPayoff,ddof=1)/np.sqrt(num_sim)
    confidence.iloc[0,2] = estimate+1.96*np.std(nPayoff,ddof=1)/np.sqrt(num_sim)
    confidence.iloc[1,0] = estimate-2.58*np.std(nPayoff,ddof=1)/np.sqrt(num_sim) 
    confidence.iloc[1,2] = estimate+2.58*np.std(nPayoff,ddof=1)/np.sqrt(num_sim)
    confidence.iloc[0,1] = estimate
    confidence.iloc[1,1] = estimate                          
    return confidence

def av_monte_carlo(num_sim,S01,S02,sigma1,sigma2,q1,q2,r,T,rho):
    nPayoff = []
    nAV_Payoff = []
    y = []
    for i in range (num_sim):
        WT1 = rd.gauss(0,1.0)
        WT2 = rd.gauss(0,1.0)
        ST1 = S01*np.exp((r-q1-0.5*sigma1**2)*T+sigma1*np.sqrt(T)*WT1)
        ST2 = S02*np.exp((r-q2-0.5*sigma2**2)*T+sigma2*rho*np.sqrt(T)*WT1+sigma2*np.sqrt(1-rho**2)*np.sqrt(T)*WT2)
        AV_ST1 = S01*np.exp((r-q1-0.5*sigma1**2)*T+sigma1*np.sqrt(T)*(-WT1))
        AV_ST2 = S02*np.exp((r-q2-0.5*sigma2**2)*T+sigma2*rho*np.sqrt(T)*(-WT1)+sigma2*np.sqrt(1-rho**2)*np.sqrt(T)*(-WT2))
        nPayoff.append(np.maximum(ST1,ST2))
        nAV_Payoff.append(np.maximum(AV_ST1,AV_ST2))
        y.append(np.exp(-r*T)*(np.maximum(ST1,ST2)+np.maximum(AV_ST1,AV_ST2))/2)
    estimate = 1/num_sim*sum(y)
    confidence = pd.DataFrame(np.zeros((2,3)), columns = ['Lower Band','Estimate','Upper Band'], index = ['95%','99%'])
    confidence.iloc[0,0] = estimate-1.96*np.std(y,ddof=1)/np.sqrt(num_sim)
    confidence.iloc[0,2] = estimate+1.96*np.std(y,ddof=1)/np.sqrt(num_sim)
    confidence.iloc[1,0] = estimate-2.58*np.std(y,ddof=1)/np.sqrt(num_sim) 
    confidence.iloc[1,2] = estimate+2.58*np.std(y,ddof=1)/np.sqrt(num_sim)
    confidence.iloc[0,1] = estimate
    confidence.iloc[1,1] = estimate                          
    return confidence

def comparison_monte_carlo(num_sim,S01,S02,sigma1,sigma2,q1,q2,r,T,rho):
    nPayoff = []
    nAV_Payoff = []
    y = []
    for i in range (num_sim):
        WT1 = rd.gauss(0,1.0)
        WT2 = rd.gauss(0,1.0)
        ST1 = S01*np.exp((r-q1-0.5*sigma1**2)*T+sigma1*np.sqrt(T)*WT1)
        ST2 = S02*np.exp((r-q2-0.5*sigma2**2)*T+sigma2*rho*np.sqrt(T)*WT1+sigma2*np.sqrt(1-rho**2)*np.sqrt(T)*WT2)
        AV_ST1 = S01*np.exp((r-q1-0.5*sigma1**2)*T+sigma1*np.sqrt(T)*(-WT1))
        AV_ST2 = S02*np.exp((r-q2-0.5*sigma2**2)*T+sigma2*rho*np.sqrt(T)*(-WT1)+sigma2*np.sqrt(1-rho**2)*np.sqrt(T)*(-WT2))
        nPayoff.append(np.exp(-r*T)*np.maximum(ST1,ST2))
        nAV_Payoff.append(np.maximum(AV_ST1,AV_ST2))
        y.append(np.exp(-r*T)*(np.maximum(ST1,ST2)+np.maximum(AV_ST1,AV_ST2))/2)
    estimate = 1/num_sim*sum(y)
    st_estimate = 1/num_sim*sum(nPayoff)
    confidence = pd.DataFrame(np.zeros((2,3)), columns = ['Lower Band','Estimate','Upper Band'], index = ['95%','99%'])
    confidence.iloc[0,0] = estimate-1.96*np.std(y,ddof=1)/np.sqrt(num_sim)
    confidence.iloc[0,2] = estimate+1.96*np.std(y,ddof=1)/np.sqrt(num_sim)
    confidence.iloc[1,0] = estimate-2.58*np.std(y,ddof=1)/np.sqrt(num_sim) 
    confidence.iloc[1,2] = estimate+2.58*np.std(y,ddof=1)/np.sqrt(num_sim)
    confidence.iloc[0,1] = estimate
    confidence.iloc[1,1] = estimate      

    st_confidence = pd.DataFrame(np.zeros((2,3)), columns = ['Lower Band','Estimate','Upper Band'], index = ['95%','99%'])
    st_confidence.iloc[0,0] = st_estimate-1.96*np.std(nPayoff,ddof=1)/np.sqrt(num_sim)
    st_confidence.iloc[0,2] = st_estimate+1.96*np.std(nPayoff,ddof=1)/np.sqrt(num_sim)
    st_confidence.iloc[1,0] = st_estimate-2.58*np.std(nPayoff,ddof=1)/np.sqrt(num_sim) 
    st_confidence.iloc[1,2] = st_estimate+2.58*np.std(nPayoff,ddof=1)/np.sqrt(num_sim)
    st_confidence.iloc[0,1] = st_estimate
    st_confidence.iloc[1,1] = st_estimate                          
    return confidence,  st_confidence

st.write("""
# Best of two Assets Option Price Calculator
This web application calculate the **best of two asset option** price easily and efficiently.""")
st.write("""The user might utilize the sidebar to modify the inputs, inserting them manually, selecting the _Manual Input_ tab.\n
If the user selects the _Real Data_ tab, the calculator automatically computes all the necessary parameters for the chosen stocks.\n
In both cases, default paremeters are already inserted to provide an example. 

""")
st.sidebar.header('User inputs')
def user_input_side():
    num_digits = int(st.sidebar.slider("Number of decimal digits ",1,10,3))
    S01 = float(st.sidebar.text_input('S01', 10))
    S02 = float(st.sidebar.text_input('S02', 20))
    sigma1 = float(st.sidebar.text_input('Volatility 1', 0.2))
    sigma2 = float(st.sidebar.text_input('Volatility 2', 0.3))
    rho = float(st.sidebar.text_input('Correlation', 0.25))
    q1 = float(st.sidebar.text_input('Dividend Yield 1', 0.02))
    q2 = float(st.sidebar.text_input('Dividend Yield 2', 0.03))
    r = float(st.sidebar.text_input('Risk free rate', 0.04))
    T = float(st.sidebar.text_input('Maturity', 3))
    #mat = st.sidebar.date_input("Maturity")
    num_sim = int(st.sidebar.slider('Num sim',100,10000, 1000))
    return num_sim,S01,S02,sigma1,sigma2,q1,q2,r,T,rho, num_digits

def user_input_side_real():
    num_digits = int(st.sidebar.slider("Number of decimal digits ",1,10,3))
    stock1=st.sidebar.text_input('S1','SNAP')
    stock2=st.sidebar.text_input('S2','TWTR')
    check_date = st.sidebar.radio('Define calculation period:',('Default: 2 Years', 'Custom'))
    if check_date == 'Custom':
        start_time = st.sidebar.date_input("Start date",datetime.date(2019, 11, 10))
        end_time = st.sidebar.date_input("Final date")
    else:
        end_time = datetime.datetime.now()
        start_time = datetime.datetime.now() - datetime.timedelta(days=2*365)
    r = float(st.sidebar.text_input('Risk free rate', 0.04))
    T = float(st.sidebar.text_input('Maturity', 3))
    if T < 0:
        st.sidebar.write("Please inset a valid maturity")
    num_sim = int(st.sidebar.slider('Num sim',100,10000, 1000))
    return stock1, stock2, r, T, num_sim, num_digits, start_time, end_time

def real_dividends (stock1, stock2, S01, S02):
    stock1 = yf.Ticker(stock1)
    stock2 = yf.Ticker(stock2)
    D1 = sum(stock1.dividends['2020'])
    D2 = sum(stock2.dividends['2020'])
    q1 = np.log(1+D1/S01)
    q2 = np.log(1+D2/S02)
    
    return q1, q2

def real_data(stock1, stock2, start_time, end_time):
    stock = stock1 + ' ' + stock2
    

    #end_time = datetime.datetime.now()
    #start_time = datetime.datetime.now() - datetime.timedelta(days=2*365)
    
    data = pdr.get_data_yahoo(stock, start_time, end_time)
    time_series1 = data['Adj Close'][stock1]
    time_series2 = data['Adj Close'][stock2]
    
    st.subheader("Price chart")
    
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
    
    q1, q2 = real_dividends (stock1, stock2, time_series1[-1], time_series2[-1])
    
    return time_series1[-1], time_series2[-1], sigma1, sigma2, rho, q1, q2

def error_plot(S01, S02, sigma1, sigma2, rho, r, T, q1, q2):
    error = []
    n_estimate = []
    up_bound = []
    down_bound =[]
    analytical = analytical_solution(S01,S02,sigma1,sigma2,q1,q2,r,T,rho)
    for num_sim in range (100,10000,100):
        confidence = monte_carlo(num_sim,S01,S02,sigma1,sigma2,q1,q2,r,T,rho)
        estimate = confidence.iloc[0,1]
        n_estimate.append(estimate)
        up_bound.append(confidence.iloc[0,2])
        down_bound.append(confidence.iloc[0,0])
        dummy = estimate - analytical
        error.append(dummy)
    return error, n_estimate, up_bound, down_bound

def av_error_plot(S01, S02, sigma1, sigma2, rho, r, T, q1, q2):
    error = []
    n_estimate = []
    up_bound = []
    down_bound =[]
    st_up_bound = []
    st_down_bound =[]
    analytical = analytical_solution(S01,S02,sigma1,sigma2,q1,q2,r,T,rho)
    for num_sim in range (100,10000,100):
        confidence, st_confidence = comparison_monte_carlo(num_sim,S01,S02,sigma1,sigma2,q1,q2,r,T,rho)
        estimate = confidence.iloc[0,1]
        n_estimate.append(estimate)
        up_bound.append(confidence.iloc[0,2])
        down_bound.append(confidence.iloc[0,0])
        st_up_bound.append(st_confidence.iloc[0,2])
        st_down_bound.append(st_confidence.iloc[0,0])
        dummy = estimate - analytical
        error.append(dummy)
    return error, n_estimate, up_bound, down_bound, st_up_bound, st_down_bound

yes = st.sidebar.selectbox("Which type of input would you like to use?", ('Manual Input', 'Real Data'))

if yes == 'Manual Input' : 
    num_sim,S01,S02,sigma1,sigma2,q1,q2,r,T,rho, num_digits = user_input_side()
    table = round(monte_carlo(num_sim,S01,S02,sigma1,sigma2,q1,q2,r,T,rho), num_digits)
    AV_table = round(av_monte_carlo(num_sim,S01,S02,sigma1,sigma2,q1,q2,r,T,rho), num_digits)
    estimate = table.iloc[0,1]
    AV_estimate = AV_table.iloc[0,1]
    analytical = round(analytical_solution(S01,S02,sigma1,sigma2,q1,q2,r,T,rho), num_digits)
    
    st.subheader('Analytical solution')
    st.write("The analytical solution is calculated with the following formula: ")
    st.latex(r'''S_X(0)=S_1(0)e^{-q_1T}N(d_1)-S_2(0)e^{-q_2T}N(d_2)+S_2(0)e^{-q_2T}''')
    st.latex(r'''d_1=\frac{ln{\frac{S_1(0)}{S_2(0)}}+(q_2-q_1+\frac{\sigma_1^2+\sigma_2^2-2\rho\sigma_1\sigma_2}{2})T}{\sqrt{(\sigma_1^2+\sigma_2^2-2\rho\sigma_1\sigma_2) T}}  ''')
    st.write("The price calculated with the closed-form solution is ", analytical)
   
    
    st.subheader('Monte Carlo estimate')
    st.write("The Monte Carlo estimate, obtained with "+str(num_sim)+" simulations, is ", estimate)
    st.write("The following table collects the estimate, its 95% confidence interval and its 99% confidence interval:", table)
    

    st.subheader('Estimation error')
    st.write("The absolute estimation error, obtained with "+str(num_sim)+" simulations, is ", round(np.abs(analytical - estimate),num_digits))
    
    st.subheader('Error convergence')
    st.write("Clicking the button below, the calculator generates a plot which shows the progressive reduction of the estimation error with the increase of simulations. It is possible to add a second plot that shows the evolution of the estimate price with its confidence interval, by selecting the relative option in the _data selection tool_.    ")
    st.write("The calculations will require some time!")
    genre = st.radio("Select data to be plotted",     ('Error', 'Price estimate', 'Error and Estimate'))
    button = st.button('Show convergence plot')
    if button: 
        if genre not in ['Error', 'Price estimate', 'Error and Estimate']:
            st.write("Select the data to be plotted")
        else:
            with st.spinner("Processing data..."):
                error, n_estimate, up_bound, down_bound = error_plot(S01, S02, sigma1, sigma2, rho, r, T, q1, q2)
            if genre == 'Error': 
                #st.line_chart(error)
                fig, ax = plt.subplots()
                ax.plot(range (100,10000,100),error)
                plt.title('Error')
                plt.xlabel('Num. Sim.')
                plt.ylabel('Value')
                plt.grid()
                plt.plot([100, 10000], [0, 0], 'r-', lw=2)
                ax.set_xlim(0, 10000)
                st.pyplot(fig)
            elif genre == 'Price estimate':
                #st.line_chart(n_estimate)
                fig, ax = plt.subplots()
                ax.plot(range (100,10000,100),n_estimate)
                plt.plot(range (100,10000,100),up_bound, 'c--')
                plt.plot(range (100,10000,100),down_bound, 'c--')
                plt.title("Price Estimate")
                plt.xlabel('Num. Sim.')
                plt.ylabel('Value')
                plt.grid()
                plt.plot([100, 10000], [analytical, analytical], 'r-', lw=1.5)
                ax.set_xlim(0, 10000)
                ax.legend(['Estimate','Confidence Interval'])
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots()
                ax.plot(range (100,10000,100),error)
                plt.title('Error')
                plt.xlabel('Num. Sim.')
                plt.ylabel('Value')
                plt.grid()
                plt.plot([100, 10000], [0, 0], 'r-', lw=2)
                ax.set_xlim(0, 10000)
                st.pyplot(fig)
                
                fig2, ax2 = plt.subplots()
                ax2.plot(range (100,10000,100),n_estimate)
                plt.plot(range (100,10000,100),up_bound, 'c--')
                plt.plot(range (100,10000,100),down_bound, 'c--')
                plt.title("Price Estimate")
                plt.xlabel('Num. Sim.')
                plt.ylabel('Value')
                plt.grid()
                plt.plot([100, 10000], [analytical, analytical], 'r-', lw=1.5)
                ax2.set_xlim(0, 10000)
                ax2.legend(['Estimate','Confidence Interval'])
                st.pyplot(fig2)
        
    st.subheader('Antithetic variates')
    st.write("Appling the antithetic variates method to the Monte Carlo estimation process, with "+str(num_sim)+" simulations, it is obtained an estimate equal to ", AV_estimate)
    st.write("The absolute estimation error is ", round(np.abs(analytical-AV_estimate),num_digits))
    st.write("The variance reduction is visible in the following table which collects the estimate, its 95% confidence interval and its 99% confidence interval:", AV_table)
    percentage = (AV_table.iloc[1,2]-AV_table.iloc[1,0]- (table.iloc[1,2]-table.iloc[1,0]))/(table.iloc[1,2]-table.iloc[1,0])
    st.write("The 99% confidence interval width is reduced by **",str(round(-percentage*100,2)),"%**.")
    
    st.subheader('Confidence interval analysis')
    st.write("Clicking the button below, the calculator generates a plot which shows the 95% confidence intervals obtained with the standard Monte Carlo method and with the **Antithetic Variate** method, computed with the same random drawn.")
    st.write("The calculations will require some time!")
    if st.button('Confidence interval comparison'):
        with st.spinner("Processing data..."):
                error, n_estimate, up_bound, down_bound, st_up_bound, st_down_bound = av_error_plot(S01, S02, sigma1, sigma2, rho, r, T, q1, q2)
        fig, ax = plt.subplots()
        #ax.plot(range (100,10000,100),n_estimate)
        plt.plot(range (100,10000,100),st_up_bound, 'g--')
        plt.plot(range (100,10000,100),up_bound, 'c--')
        plt.plot(range (100,10000,100),down_bound, 'c--')
        plt.plot(range (100,10000,100),st_down_bound, 'g--')
        plt.title("95% Confidence intervals")
        plt.xlabel('Num. Sim.')
        plt.ylabel('Value')
        plt.grid()
        plt.plot([100, 10000], [analytical, analytical], 'r-', lw=1.5)
        ax.set_xlim(0, 10000)
        ax.legend(['Standard MC','Antithetic Variates'])
        st.pyplot(fig)

            
else:
    stock1, stock2, r, T, num_sim, num_digits, start_time, end_time = user_input_side_real()
    S01, S02, sigma1, sigma2, rho, q1, q2 = real_data(stock1, stock2, start_time, end_time)
    
    param = pd.DataFrame(np.zeros((3,2)), columns = [stock1,stock2], index = ['Initial Value','Annual Volatility','Annual Dividend Yield'])
    param.iloc[0,0] = S01
    param.iloc[0,1] = S02
    param.iloc[1,0] = sigma1
    param.iloc[1,1] = sigma2
    param.iloc[2,0] = q1
    param.iloc[2,1] = q2
    
    st.subheader("Parameters")
    st.write("In the following table are collected the parameters calculated using the real data for the selected time frame: ", param)
    st.write("The correlation between the two stocks is **"+str(round(rho,num_digits))+"**.")
    
    table = round(monte_carlo(num_sim,S01,S02,sigma1,sigma2,q1,q2,r,T,rho), num_digits)
    AV_table = round(av_monte_carlo(num_sim,S01,S02,sigma1,sigma2,q1,q2,r,T,rho), num_digits)
    estimate = table.iloc[0,1]
    AV_estimate = AV_table.iloc[0,1]
    analytical = round(analytical_solution(S01,S02,sigma1,sigma2,q1,q2,r,T,rho), num_digits)
    
    st.subheader('Analytical solution')
    st.write("The analytical solution is calculated with the following formula: ")
    st.latex(r'''S_X(0)=S_1(0)e^{-q_1T}N(d_1)-S_2(0)e^{-q_2T}N(d_2)+S_2(0)e^{-q_2T}''')
    st.latex(r'''d_1=\frac{ln{\frac{S_1(0)}{S_2(0)}}+(q_2-q_1+\frac{\sigma_1^2+\sigma_2^2-2\rho\sigma_1\sigma_2}{2})T}{\sqrt{(\sigma_1^2+\sigma_2^2-2\rho\sigma_1\sigma_2) T}}  ''')
    st.write("The price calculated with the closed-form solution is ", analytical)
   
    
    st.subheader('Monte Carlo estimate')
    st.write("The Monte Carlo estimate, obtained with "+str(num_sim)+" simulations, is ", estimate)
    st.write("The following table collects the estimate, its 95% confidence interval and its 99% confidence interval:", table)
    

    st.subheader('Estimation error')
    st.write("The absolute estimation error, obtained with "+str(num_sim)+" simulations, is ", round(np.abs(analytical - estimate),num_digits))
    
    st.subheader('Error convergence')
    st.write("Clicking the button below, the calculator generates a plot which shows the progressive reduction of the estimation error with the increase of simulations. It is possible to add a second plot that shows the evolution of the estimate price with its confidence interval, by selecting the relative option in the _data selection tool_.    ")
    st.write("The calculations will require some time!")
    genre = st.radio("Select data to be plotted",     ('Error', 'Price estimate', 'Error and Estimate'))
    button = st.button('Show convergence plot')
    if button: 
        if genre not in ['Error', 'Price estimate', 'Error and Estimate']:
            st.write("Select the data to be plotted")
        else:
            with st.spinner("Processing data..."):
                error, n_estimate, up_bound, down_bound = error_plot(S01, S02, sigma1, sigma2, rho, r, T, q1, q2)
            if genre == 'Error': 
                #st.line_chart(error)
                fig, ax = plt.subplots()
                ax.plot(range (100,10000,100),error)
                plt.title('Error')
                plt.xlabel('Num. Sim.')
                plt.ylabel('Value')
                plt.grid()
                plt.plot([100, 10000], [0, 0], 'r-', lw=2)
                ax.set_xlim(0, 10000)
                st.pyplot(fig)
            elif genre == 'Price estimate':
                #st.line_chart(n_estimate)
                fig, ax = plt.subplots()
                ax.plot(range (100,10000,100),n_estimate)
                plt.plot(range (100,10000,100),up_bound, 'c--')
                plt.plot(range (100,10000,100),down_bound, 'c--')
                plt.title("Price Estimate")
                plt.xlabel('Num. Sim.')
                plt.ylabel('Value')
                plt.grid()
                plt.plot([100, 10000], [analytical, analytical], 'r-', lw=1.5)
                ax.set_xlim(0, 10000)
                ax.legend(['Estimate','Confidence Interval'])
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots()
                ax.plot(range (100,10000,100),error)
                plt.title('Error')
                plt.xlabel('Num. Sim.')
                plt.ylabel('Value')
                plt.grid()
                plt.plot([100, 10000], [0, 0], 'r-', lw=2)
                ax.set_xlim(0, 10000)
                st.pyplot(fig)
                
                fig2, ax2 = plt.subplots()
                ax2.plot(range (100,10000,100),n_estimate)
                plt.plot(range (100,10000,100),up_bound, 'c--')
                plt.plot(range (100,10000,100),down_bound, 'c--')
                plt.title("Price Estimate")
                plt.xlabel('Num. Sim.')
                plt.ylabel('Value')
                plt.grid()
                plt.plot([100, 10000], [analytical, analytical], 'r-', lw=1.5)
                ax2.set_xlim(0, 10000)
                ax2.legend(['Estimate','Confidence Interval'])
                st.pyplot(fig2)
        
    st.subheader('Antithetic variates')
    st.write("Appling the antithetic variates method to the Monte Carlo estimation process, with "+str(num_sim)+" simulations, it is obtained an estimate equal to ", AV_estimate)
    st.write("The absolute estimation error is ", round(np.abs(analytical-AV_estimate),num_digits))
    st.write("The variance reduction is visible in the following table which collects the estimate, its 95% confidence interval and its 99% confidence interval:", AV_table)
    percentage = (AV_table.iloc[1,2]-AV_table.iloc[1,0]- (table.iloc[1,2]-table.iloc[1,0]))/(table.iloc[1,2]-table.iloc[1,0])
    st.write("The 99% confidence interval width is reduced by **",str(round(-percentage*100,2)),"%**.")
    
    st.subheader('Confidence interval analysis')
    st.write("Clicking the button below, the calculator generates a plot which shows the 95% confidence intervals obtained with the standard Monte Carlo method and with the **Antithetic Variate** method, computed with the same random drawn.")
    st.write("The calculations will require some time!")
    if st.button('Confidence interval comparison'):
        with st.spinner("Processing data..."):
                error, n_estimate, up_bound, down_bound, st_up_bound, st_down_bound = av_error_plot(S01, S02, sigma1, sigma2, rho, r, T, q1, q2)
        fig, ax = plt.subplots()
        #ax.plot(range (100,10000,100),n_estimate)
        plt.plot(range (100,10000,100),st_up_bound, 'g--')
        plt.plot(range (100,10000,100),up_bound, 'c--')
        plt.plot(range (100,10000,100),down_bound, 'c--')
        plt.plot(range (100,10000,100),st_down_bound, 'g--')
        plt.title("95% Confidence intervals")
        plt.xlabel('Num. Sim.')
        plt.ylabel('Value')
        plt.grid()
        plt.plot([100, 10000], [analytical, analytical], 'r-', lw=1.5)
        ax.set_xlim(0, 10000)
        ax.legend(['Standard MC','Antithetic Variates'])
        st.pyplot(fig)



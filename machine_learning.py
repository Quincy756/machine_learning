# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 11:22:12 2020

@author: quenc
"""

import  pymysql
import  matplotlib.pyplot as plt
import  numpy as np


connection = pymysql.connect(
    host = "localhost",
    user = "root",
    password = "tangfeng75609378",
    port = 3306,
    db = "sz_sechandhouse_db",
    charset="utf8"
)
cursor = connection.cursor()


def plt_price_area(price, area):
    
    # xticks = np.linspace(0, 1000, 101) 
    # yticks = np.linspace(0,200000)
    plt.scatter(area[0:100], price[0:100])
    plt.grid(True) 
    plt.xlabel('面积/$m^2$',fontproperties='SimHei')
    plt.ylabel('价格/元',fontproperties='SimHei')
    plt.title('深圳某地区房价与面积关系散点图',fontproperties='SimHei')
    # plt.xticks()
    # plt.yticks()
    plt.show()

def regression(x, y):
    
    x_1tm = np.array(x)
    y_1tm = np.array(y) 
    #特征缩放
    # print(x_1tm.shape)
    x_1tm = ((x_1tm-x_1tm.min())/(x_1tm.max()-x_1tm.min())).reshape(x_1tm.shape[0],1)
    Y = ((y_1tm-y_1tm.min())/(y_1tm.max()-y_1tm.min())).reshape(y_1tm.shape[0],1)

    x_0 = np.ones((x_1tm.shape[0],1),np.float32) 
    X = np.hstack((x_0, x_1tm))
    
    num = X.shape[0]
    theta = np.array([0,1]).reshape(2,1)
    print(theta.shape)
    print(X.shape)
    print(Y.shape)
    alpha = 1
    
    J1 = np.array([])
    for i in  range(200):
        J = np.dot(np.transpose(X),(np.dot(X,theta)-Y))
        # print(J)
        J1 = np.append(J1,np.sum(J))
        # print(J.shape)
        theta = theta- (alpha/num)*J
        # print(theta.shape)
    H = np.dot(X, theta)
    print(J1.shape)
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure()
    plt.plot(X[:,1],H,label='拟合线',linestyle='-',color='c')
    plt.scatter(X[:,1],Y,label='实际散点')
    plt.grid(True) 
    plt.xlabel('面积/$m^2$')
    plt.ylabel('价格/元')
    plt.title('深圳某地区房价与面积关系拟合图')
    plt.legend()
    plt.figure()
    plt.grid(True)
    plt.plot(range(200),J1)
    plt.xlabel('迭代次数')
    plt.ylabel('误差')
    plt.title('误差变化曲线')
    # plt.title('拟合误差变化曲线')
    
    # print(the)
    # plt.scatter(range(-2,2,0.01),J)
    # plt.plot(theta_0_list,J,'r-')
    plt.show()

def cost(x, y):
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    x1 = (x-x.min())/(x.max()-x.min())
    x0 = np.ones_like(x1)
    
    X = np.vstack((x0, x1))
    # print('X shape is', X.shape)
    m = X.shape[1]
    
    y = (y-y.min())/(y.max()-y.min())               
    y = np.reshape(y, (1, y.shape[0]))
    # theta1 = np.linspace(-10,10,200)
    # theta0 = np.zeros_like(theta1)
    
    # theta = np.vstack((theta0, theta1))
    # print(theta.shape)
    
    # H = np.dot(np.transpose(theta), X)
    # # J = 0.5m*np.sum((H-y))
    # J = np.array([])
    
    #不同学习率
    theta = np.array([[0],[0.2]])
    J1 = np.array([])
    alpha = 1
    print(y.shape) 
    theta_list = np.array([])
    for alpha in np.linspace(0.6, 2, 5):
        theta_list = np.array([])
        theta = np.array([[0],[0.2]])
        J1 = np.array([])
        for i in range(60):
               
                theta_list = np.append(theta_list, theta)
                h = np.dot(np.transpose(theta), X)
                # print(h.shape)
                j = (0.5/m)*np.sum((h-y)**2)
                # print(j)
                J1 = np.append(J1, j)
                # print((h-y).shape)
                dJ = np.dot((h-y), np.transpose(X))
                theta = theta - (alpha/m)*np.transpose(dJ)
                # print(theta.shape)        
                # print(theta) 
                # x_y = plt.subplot(121)     # 
                # x_y.scatter(x1, y, color = 'c',label= 'original')       # theta = () 
                # xx1 = np.linspace(0,1,101)
                # xx0 = np.ones_like(xx1)
                # xx = np.vstack((xx0, xx1)) 
                # print(theta.shape)
                # x_y.plot(xx[1,:], (np.dot(np.transpose(theta), xx)).flatten(), color='r', label='result', linewidth=3)
                # plt.legend(loc='upper left')
                # x_y.set_title('regression')
               
        plt.plot(J1,label='learning rate ={0:.2f}'.format(alpha))
        plt.grid(True)
        plt.legend()
    plt.xlabel('迭代次数')
    plt.ylabel('误差')
    plt.title('不同学习率下误差变化曲线')
    plt.show()
    
    
    
    
    


try:   
    print('连接成功') 
    sql = 'select total_price,area from other_tb'
    cursor.execute(sql)
    result = cursor.fetchall()
    price = [result[num][0]  for num in range(len(result))]
    area = [result[num][1]  for num in range(len(result))]
    cost(area[0:100], price[0:100])
    # print(price)
    # print(area)
    # plt_price_area(price, area)   
    # regression(area[0:100], price[0:100])  
    
    
    
    
    
    
    
    
    
    
    
    
except Exception as error:
    print('the error is', error)
    


    
    
    
    
    
    
    
    
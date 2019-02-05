# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 13:42:57 2018

@author: p007031B
"""

import pandas as pd  # data frame operations  
import matplotlib.pyplot as plt  # static plotting


data1 = pd.read_csv('incident_pred_1117890.csv')
data2 = pd.read_csv('incident_pred_1117890.csv')
data1['Timestamp'] = pd.to_datetime(data1['Timestamp'])




y_Flow = ["VDS_Lane_1_Flow","VDS_Lane_2_Flow","VDS_Lane_3_Flow","VDS_Lane_4_Flow"]

y_Occupancy = ["VDS_Lane_1_Occupancy","VDS_Lane_2_Occupancy","VDS_Lane_3_Occupancy",
               "VDS_Lane_4_Occupancy","VDS_Lane_5_Occupancy","VDS_Lane_6_Occupancy",
               "VDS_Lane_7_Occupancy","VDS_Lane_8_Occupancy"]

color2 = "0.5"

plt.style.use('dark_background')


fig = plt.figure()
ax=fig.add_subplot(111, label="1")
ax2=fig.add_subplot(111, label="2", frame_on=False)

ax.plot(data1['Timestamp'],data1[y_Flow],zorder=100)
ax.set_xlabel("Time", color="k")
ax.set_ylabel("Flow (Veh/Min)", color="C0")
ax.tick_params(axis='x', colors="k")
ax.tick_params(axis='y', colors="C0")
ax.xaxis.grid()
ax.yaxis.grid()
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,
           labels = y_Flow)

ax2.bar(data2['Timestamp'],data2['isIncident'],width=0.3, color=color2,zorder=0)
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.set_ylabel('Incident Probability', color=color2)       
ax2.xaxis.set_label_position('top') 
ax2.axes.get_xaxis().set_visible(False)
ax2.yaxis.set_label_position('right') 
ax2.tick_params(axis='x', colors=color2)
ax2.tick_params(axis='y', colors="k")

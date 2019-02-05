# Cody Wittenkeller MSDS 422 Assignment 1

# Solutions created were with the assistance of:
# Jump-Start Example: Python analysis of MSPA Software Survey
# Update 2017-09-21 by Tom Miller and Kelsey O'Neill

# visualizations in this program are routed to external pdf files
# so they may be included in printed or electronic reports

# external libraries for visualizations and data manipulation
# ensure that these packages have been installed prior to calls
import pandas as pd  # data frame operations  
import numpy as np  # arrays and math functions
import matplotlib.pyplot as plt  # static plotting
import seaborn as sns  # pretty plotting, including heat map

# read in comma-delimited text file, creating a pandas DataFrame object
# note that IPAddress is formatted as an actual IP address
# but is actually a random-hash of the original IP address
valid_survey_input = pd.read_csv('mspa-survey-data.csv')

# use the RespondentID as label for the rows... the index of DataFrame
valid_survey_input.set_index('RespondentID', drop = True, inplace = True)

# examine the structure of the DataFrame object
print('\nContents of initial survey data= {}'.format(valid_survey_input.shape))

# show the column/variable names of the DataFrame
# note that RespondentID is no longer present
print(valid_survey_input.columns)

# abbreviated printing of the first five rows of the data frame
print(pd.DataFrame.head(valid_survey_input)) 

# shorten the variable/column names for software preference variables
survey_df = valid_survey_input.rename(index=str, columns={
    'Personal_JavaScalaSpark': 'My_Java',
    'Personal_JavaScriptHTMLCSS': 'My_JS',
    'Personal_Python': 'My_Python',
    'Personal_R': 'My_R',
    'Personal_SAS': 'My_SAS',
    'Professional_JavaScalaSpark': 'Prof_Java',
    'Professional_JavaScriptHTMLCSS': 'Prof_JS',
    'Professional_Python': 'Prof_Python',
    'Professional_R': 'Prof_R',
    'Professional_SAS': 'Prof_SAS',
    'Industry_JavaScalaSpark': 'Ind_Java',
    'Industry_JavaScriptHTMLCSS': 'Ind_JS',
    'Industry_Python': 'Ind_Python',
    'Industry_R': 'Ind_R',
    'Industry_SAS': 'Ind_SAS'})

# define subset DataFrame for analysis of software preferences 
software_df = survey_df.loc[:, 'My_Java':'Ind_SAS']
      
survey_df_labels = [
    'Personal Preference for Java/Scala/Spark',
    'Personal Preference for Java/Script/HTML/CSS',
    'Personal Preference for Python',
    'Personal Preference for R',
    'Personal Preference for SAS',
    'Professional Java/Scala/Spark',
    'Professional JavaScript/HTML/CSS',
    'Professional Python',
    'Professional R',
    'Professional SAS',
    'Industry Java/Scala/Spark',
    'Industry Java/Script/HTML/CSS',
    'Industry Python',
    'Industry R',
    'Industry SAS'        
]    


# descriptive statistics for software preference variables     
# allows for general statistics to be viewed and compared for decision making

print(valid_survey_input.describe())

# create a set of boxplots for all software preferences by courses completed
# boxplots allow for median, IQR, range, and outliers to be visualized
# grouping by courses completed allows for a better understanding of the 
# relationship between time spent in the program and perception of softwares

for i in range(0,19):
            file_title = survey_df.columns[i] + '_by Courses_Completed'
            plot_title = survey_df.columns[i] + ' by Courses Completed'
            fig, axis = plt.subplots()
            plt.title(plot_title)
            sns.boxplot(x="Courses_Completed",y=survey_df.columns[i],data=survey_df)
            plt.savefig(file_title + '.pdf', 
                bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
                orientation='portrait', papertype=None, format=None, 
                transparent=True, pad_inches=0.25, frameon=None) 

# correlation heat map setup for seaborn
# creates a correlation heat map based on the software preferences amongst students
# strongest correlations exist between personal preference and professional and
# industry preferences.

def corr_chart(df_corr):
    corr=df_corr.corr()
    #screen top half to get a triangle
    top = np.zeros_like(corr, dtype=np.bool)
    top[np.triu_indices_from(top)] = True
    fig=plt.figure()
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(corr, mask=top, cmap='coolwarm', 
        center = 0, square=True, 
        linewidths=.5, cbar_kws={'shrink':.5}, 
        annot = True, annot_kws={'size': 9}, fmt = '.3f')           
    plt.xticks(rotation=90) # rotate variable labels on columns (x axis)
    plt.yticks(rotation=0) # use horizontal variable labels on rows (y axis)
    plt.title('Correlation Heat Map')   
    plt.savefig('plot-corr-map.pdf', 
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
        orientation='portrait', papertype=None, format=None, 
        transparent=True, pad_inches=0.25, frameon=None)      

np.set_printoptions(precision=3)
corr_chart(df_corr = software_df) 


# Seaborn provides a convenient way to show the effects of transformations
# on the distribution of values being transformed
# Documentation at https://seaborn.pydata.org/generated/seaborn.distplot.html
#python and R are the softwares of highest interest ratings
#the My_R and My_Python values will be explored through transofrmations

# the transformations do not indicate any significant difference in the
# variables for all of the different representations.

#import libraries necessary
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

X_R = np.array(survey_df['My_R'].dropna()).reshape(-1,1)
X_Python= np.array(survey_df['My_Python'].dropna()).reshape(-1,1)

unscaled_fig, ax = plt.subplots()
sns.distplot(X_R,kde_kws={"color": "b", "lw": 2, "label": "My_R"}).set_title('Unscaled')
sns.distplot(X_Python,kde_kws={"color": "orange", "lw": 2, "label": "My_Python"}).set_title('Unscaled')
unscaled_fig.savefig('Transformation-Unscaled' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)  

standard_fig, ax = plt.subplots()
sns.distplot(StandardScaler().fit_transform(X_R),
             kde_kws={"color": "b", "lw": 2, "label": "My_R"}).set_title('StandardScaler')
sns.distplot(StandardScaler().fit_transform(X_Python),
             kde_kws={"color": "orange", "lw": 2, "label": "My_Python"}).set_title('StandardScaler')
standard_fig.savefig('StandardScaler' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)  

minmax_fig, ax = plt.subplots()
sns.distplot(MinMaxScaler().fit_transform(X_R),
             kde_kws={"color": "b", "lw": 2, "label": "My_R"}).set_title('MinMaxScaler')
sns.distplot(MinMaxScaler().fit_transform(X_Python),
             kde_kws={"color": "orange", "lw": 2, "label": "My_Python"}).set_title('MinMaxScaler')
minmax_fig.savefig('MinMaxScaler' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)  

RobustScaler_fig, ax = plt.subplots()
sns.distplot(RobustScaler(quantile_range=(25, 75)).fit_transform(X_R),
             kde_kws={"color": "b", "lw": 2, "label": "My_R"}).set_title('RobustScaler')
sns.distplot(RobustScaler(quantile_range=(25, 75)).fit_transform(X_Python),
             kde_kws={"color": "orange", "lw": 2, "label": "My_Python"}).set_title('RobustScaler')
RobustScaler_fig.savefig('RobustScaler' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)  

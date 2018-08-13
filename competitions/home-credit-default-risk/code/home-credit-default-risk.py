import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')

bureau = pd.read_csv('../input/bureau.csv')
bureau.head()

#Groupby the client id (SK_ID_CURR), count hte number of loans and rename column
previous_loan_count = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU']\
    .count().rename(columns = { 'SK_ID_BUREAU' : 'previous_loan_counts'})
previous_loan_count.head()

#Join th the training dataframe
train = pd.read_csv('../input/application_train.csv')
train = train.merge(previous_loan_count, on = 'SK_ID_CURR', how = 'left')

# Fill the missing values with 0
train['previous_loan_counts'] = train['previous_loan_counts'].fillna(0)


def kde_target(var_name, df):
    corr = df['TARGET'].corr(df[var_name])

    avg_repaid = df.ix[df['TARGET'] == 0, var_name].median()
    avg_not_repaid = df.ix[df['TARGET'] == 1, var_name].median()

    plt.figure(figsize = (12, 6))

    df[var_name] = df[var_name].fillnan(0)

    sns.kdeplot(df.ix[df['TARGET'] == 0, var_name], label = 'TARGET == 0')
    sns.kdeplot(df.ix[df['TARGET'] == 1, var_name], label = 'TARGET == 1')

    plt.xlabel(var_name)
    plt.ylabel('Density')
    plt.title('%s Distribution' % var_name)
    plt.legend()

    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    print('Median value for loan that was not repaid =  %0.4f' % avg_not_repaid)
    print('Median value for loan that was repaid %0.4f' % avg_repaid)


kde_target('EXT_SOURCE_3', train)

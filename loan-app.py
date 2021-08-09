import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import random

data_info = pd.read_csv('Data/lending_club_info.csv', index_col='LoanStatNew')


@st.cache
def feat_info(col_name):
    return(data_info.loc[col_name]['Description'])


@st.cache(allow_output_mutation=True)
def get_data():
    return pd.read_csv('Data/lending_club_loan_two.csv')


df = get_data()

st.title("""
Loan Repayment Classification
""")

st.header('Our Goal')

st.write("""Given historical data on loans given out with information
 on whether or not the borrower defaulted (charge-off), can we build a
 model that can predict wether or nor a borrower will pay back their loan.
 This way in the future when we get a new potential customer we can assess
 whether or not they are likely to pay back the loan.
""")

st.header('The Data')
st.write("""We will be using a subset of the LendingClub DataSet obtained
 from Kaggle: https://www.kaggle.com/wordsforthewise/lending-club""")

st.subheader('Data Overview')

st.write("""
* *loan_amnt* -	The listed amount of the loan applied for by the borrower.
 If at some point in time, the credit department reduces the loan amount,
 then it will be reflected in this value.
* *term* -	The number of payments on the loan. Values are in months and can be
 either 36 or 60.
* *int_rate* -	Interest Rate on the loan
* *installment* - Monthly payment owed by the borrower if the loan originates.
* *grade* -	LC assigned loan grade
* *sub_grade* -	LC assigned loan subgrade
* *emp_title* -	The job title supplied by the Borrower when applying for
 the loan.
* *emp_length* -	Employment length in years. Possible values are between
 0 and 10 where 0 means less than one year and 10 means ten or more years.
* *home_ownership* -	The home ownership status provided by the borrower during
 registration or obtained from the credit report. Our values are:
 RENT, OWN, MORTGAGE, OTHER.
* *annual_inc* -	The self-reported annual income provided by the borrower
 during registration.
* *verification_status* -	Indicates if income was verified by LC, not verified,
 or if the income source was verified
* *issue_d* -	The month which the loan was funded
* *loan_status* -	Current status of the loan
* *purpose* -	A category provided by the borrower for the loan request.
* *title* -	The loan title provided by the borrower
* *zip_code* -	The first 3 numbers of the zip code provided by the borrower
 in the loan application.
* *addr_state* -	The state provided by the borrower in the loan application
* *dti*	- A ratio calculated using the borrower’s total monthly debt payments
 on the total debt obligations, excluding mortgage and the requested
 LC loan, divided by the borrower’s self-reported monthly income.
* *earliest_cr_line* -	The month the borrower's earliest reported credit
 line was opened
* *open_acc* - The number of open credit lines in the borrower's credit file.
* *pub_rec*	Number of derogatory public records
* *revol_bal* -	Total credit revolving balance
* *revol_util* -	Revolving line utilization rate, or the amount of credit
 the borrower is using relative to all available revolving credit.
* *total_acc* -	The total number of credit lines currently in the borrower's
 credit file.
* *initial_list_status* -	The initial listing status of the loan.
 Possible values are – W, F.
* *application_type* -	Indicates whether the loan is an individual
 application or a joint application with two co-borrowers
* *mort_acc* -	Number of mortgage accounts.
* *pub_rec_bankruptcies* -	Number of public record bankruptcies """)

st.subheader('Data Info')

buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.header('Section 1: Exploratory Data Analysis')
st.write("""Let's create a countplot of the loan_status column since we
 are attempting to predict it.""")
fig1, ax1 = plt.subplots()
sns.countplot(x='loan_status', data=df, ax=ax1)
st.pyplot(fig1)

st.write("""A distplot of the loan_amnt column to see the
 distribution of the column.""")
fig2, ax2 = plt.subplots(figsize=(12, 4))
sns.distplot(df['loan_amnt'], kde=False, bins=40, ax=ax2)
plt.xlim(0, 45000)
st.pyplot(fig2)

st.write("""Let's calculate the correlation between the numerical columns.""")
st.dataframe(df.corr())

st.write("""Visualise the correlation using a heatmap.""")
fig3, ax3 = plt.subplots(figsize=(12, 7))
sns.heatmap(df.corr(), annot=True, cmap='viridis', ax=ax3)
plt.ylim(10, 0)
st.pyplot(fig3)

st.write("""There is almost a perfect positive correlation with the
 installment feature. Time explore that further.
 * *firstly let's print out the description of the features*
 """)
st.write('* installment')
st.text(feat_info('installment'))

st.write('* loan_amnt')
st.text(feat_info('loan_amnt'))

st.write("""Seems like the two features are somewhat related by some formula
 used by the company to calculate the monthly installment on a given loan.
 * *Next, lets's create a scatter plot for these features.*""")
fig4, ax4 = plt.subplots()
sns.scatterplot(x='installment', y='loan_amnt', data=df, ax=ax4)
st.pyplot(fig4)

st.write("""We will use a boxplot to show the relationship between the
 loan_amnt and loan_status feature to see if there is a relationship between
 having low amount loan and paying it off or having a high amount loan
 and not being able to pay off""")
fig5, ax5 = plt.subplots()
sns.boxplot(x='loan_status', y='loan_amnt', data=df, ax=ax5)
st.pyplot(fig5)

st.write("""In general looks that they are pretty similar so doesnt really show
 wether someone will payback their loans although charged_off is a little
 higher which makes sense because the higher the the more likely it won't
 be fully paid.""")

st.write("""We will now explore the grade and subgrade feature that the company
attribute to the loans.

a count plot per grade with the hue set as the loan_status""")
fig6, ax6 = plt.subplots()
sns.countplot(x='grade', data=df, hue='loan_status', ax=ax6)
st.pyplot(fig6)

st.write("""Now, let's do same for the subgrade feature""")
fig7, ax7 = plt.subplots(figsize=(12, 4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade', data=df, order=subgrade_order,
              palette='coolwarm', ax=ax7, hue='loan_status')
st.pyplot(fig7)

st.write("""F and G subgrades don't get paid back that often.
 Let's create a countplot of these subgrades. """)
f_and_g = df[(df['grade'] == 'G') | (df['grade'] == 'F')]

fig8, ax8 = plt.subplots(figsize=(12, 4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade', data=f_and_g, order=subgrade_order,
              hue='loan_status', ax=ax8)
st.pyplot(fig8)

st.write("""This shows that the percentage of charged_off and fully_paid
 is almost similar with these subgrades meaning there is a higher chance
 of loans from these subgrades being charged_off. """)
st.markdown("""Create a new column called 'load_repaid' which will contain a 1
 if the loan status was "Fully Paid" and a 0 if it was "Charged Off".""")

df['loan_repaid'] = df['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})
st.dataframe(df[['loan_repaid', 'loan_status']])

st.header('Section 2: Data PreProcessing')
st.subheader('Section Goals:')
st.write("""Remove or fill any missing data. Remove unnecessary or repetitive
 features. Convert categorical string features to dummy variables.

 """)

st.text(df.isnull().sum())

st.markdown("""Let's examine emp_title and emp_length to see whether it will be
 okay to drop them.""")

st.markdown("* emp_title description")
st.text(feat_info('emp_title'))

st.markdown("* emp_length description")
st.text(feat_info('emp_length'))

st.text(df['emp_title'].value_counts())
st.markdown("""Realistically there are too many unique job titles to try to
 convert this to a dummy variable feature. So drop the feature.""")

df = df.drop('emp_title', axis=1)

st.markdown("""A count plot of the emp_length column.""")

sorted(df['emp_length'].dropna().unique())
emp_length_order = ['< 1 year',
                    '1 year',
                    '2 years',
                    '3 years',
                    '4 years',
                    '5 years',
                    '6 years',
                    '7 years',
                    '8 years',
                    '9 years',
                    '10+ years']

fig9, ax9 = plt.subplots(figsize=(12, 4))
sns.countplot(x='emp_length', data=df, order=emp_length_order, ax=ax9)
st.pyplot(fig9)

st.markdown("""Next We will plot out the countplot with a hue separating Fully
Paid vs Charged Off.""")

fig10, ax10 = plt.subplots(figsize=(12, 4))
sns.countplot(x='emp_length', data=df, order=emp_length_order, ax=ax10,
              hue='loan_status')
st.pyplot(fig10)

st.markdown("""This still doesn't really inform us if there is a strong
 relationship between employment length and being charged off, what we want
 is the percentage of charge offs per category. Essentially informing us what
 percentage of people per employment category didn't pay back their loan.""")

emp_co = df[df['loan_status'] == "Charged Off"].groupby("emp_length").count()[
    'loan_status']

emp_fp = df[df['loan_status'] == "Fully Paid"].groupby("emp_length").count()[
    'loan_status']

emp_len = emp_co / emp_fp

st.text(emp_len)

fig11, ax11 = plt.subplots()
emp_len.plot(kind='bar', ax=ax11)
st.pyplot(fig11)

st.markdown("""Charge off rates are extremely similar across all employment
 lengths. So we will drop this column. """)

df = df.drop('emp_length', axis=1)

st.markdown("""Review the title column vs the purpose column.""")

st.dataframe(df['purpose'].head(10))
st.dataframe(df['title'].head(10))

st.markdown("""

 The title column is simply a string subcategory/description of
 the purpose column. We will go ahead and drop the title column.
 """)

df = df.drop('title', axis=1)

st.markdown("Let's take a look at the mort_acc column, first the description")
st.markdown("mort_acc description")
st.text(feat_info('mort_acc'))

st.text(df['mort_acc'].value_counts())

st.markdown(""" There are many ways we could deal with this missing data.
 We could attempt to build a simple model to fill it in, such as a linear
 model, we could just fill it in based on the mean of the other columns,
 or you could even bin the columns into categories and then set NaN as its
 own category. Let's review the other columns to see which most highly
 correlates to mort_acc""")

st.markdown("Correlation with the mort_acc column")
st.text(df.corr()['mort_acc'].sort_values())

st.markdown("""Looks like the total_acc feature correlates with the mort_acc,
 this makes sense! Let's try this fillna() approach. We will group the
 dataframe by the total_acc and calculate the mean value for the mort_acc
 per total_acc entry. To get the result below:""")

st.markdown("Mean of mort_acc column per total_acc")
st.text(df.groupby('total_acc').mean()['mort_acc'])

st.markdown("""Let's fill in the missing mort_acc values based on their
 total_acc value. If the mort_acc is missing, then we will fill in that
 missing value with the mean value corresponding to its total_acc value from
 the series created above. This involves using an .apply() method
 with two columns.""")

total_acc_avg = df.groupby('total_acc').mean()['mort_acc']


def fill_mort_acc(total_acc, mort_acc):
    '''
    Accepts the total_acc and mort_acc values for the row.
    Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value
    for the corresponding total_acc value for that row.

    total_acc_avg here should be a Series or dictionary containing the mapping
     of the
    groupby averages of mort_acc per total_acc values.
    '''
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


df['mort_acc'] = df.apply(lambda x: fill_mort_acc(
    x['total_acc'], x['mort_acc']), axis=1)

st.text(df.isnull().sum())

st.markdown("""revol_util and the pub_rec_bankruptcies have missing data
 points, but they account for less than 0.5% of the total data. We will go
 ahead and remove the rows that are missing those values in those columns
 with dropna().""")

df = df.dropna()
st.text(df.isnull().sum())

st.markdown("""We're done working with the missing data! Now we just need to
 deal with the string values due to the categorical columns. """)

st.header("Categorical Variables and Dummy Variables")

st.markdown("""Let's check the non-numeric columns and see what we can do
with them""")

st.text(df.select_dtypes(['object']).columns)

st.subheader('term feature')

st.text(df['term'].value_counts())

st.markdown("""Let's convert the term feature into either a 36 or 60 integer
 numeric data type using .apply()""")

df['term'] = df['term'].apply(lambda term: int(term[:3]))

st.subheader('grade feature')
st.markdown("""We already know grade is part of sub_grade, so just drop the
 grade feature.
 Let's convert the subgrade into dummy variables. Then concatenate these
 new columns to the original dataframe.""")

df = df.drop('grade', axis=1)
subgrade_dummies = pd.get_dummies(df['sub_grade'], drop_first=True)
df = pd.concat([df.drop('sub_grade', axis=1), subgrade_dummies], axis=1)
st.text(df.columns)

st.subheader(
    """verification_status, application_type,initial_list_status,purpose
     features""")
st.markdown("""We will convert these columns into dummy variables and
 concatenate them with the original dataframe.""")

dummies = pd.get_dummies(df[['verification_status', 'application_type',
                         'initial_list_status', 'purpose']], drop_first=True)
df = df.drop(['verification_status', 'application_type',
             'initial_list_status', 'purpose'], axis=1)
df = pd.concat([df, dummies], axis=1)

st.subheader('home_ownership feature')
st.text(df['home_ownership'].value_counts())
st.markdown("""Let's convert these to dummy variables, but replace NONE and
 ANY with OTHER, so that we end up with just 4 categories, MORTGAGE, RENT,
 OWN, OTHER.""")

df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

dummies = pd.get_dummies(df['home_ownership'], drop_first=True)
df = df.drop('home_ownership', axis=1)
df = pd.concat([df, dummies], axis=1)

st.subheader('address feature')
st.markdown("""Let's create a zip code column from the address in the data set.
 """)

df['zip_code'] = df['address'].apply(lambda address: address[-5:])
st.markdown("Now we wil make this zip_code column into dummy variables.")

dummies = pd.get_dummies(df['zip_code'], drop_first=True)
df = df.drop(['zip_code', 'address'], axis=1)
df = pd.concat([df, dummies], axis=1)

st.subheader('issue_d feature')
st.markdown("""This would be data leakage, we wouldn't know beforehand whether
 or not a loan would be issued when using our model, so in theory we wouldn't
 have an issue_date, we will drop this feature.
 """)
df = df.drop('issue_d', axis=1)

st.subheader('earliest_cr_line feature')
st.markdown("""This appears to be a historical timestamp. We will extract the
year from the column and set it to a new column.
 """)

df['earliest_cr_year'] = df['earliest_cr_line'].apply(
    lambda date: int(date[-4:]))
df = df.drop('earliest_cr_line', axis=1)

st.text(df.select_dtypes(['object']).columns)

st.subheader('Train Test Split')
st.markdown("""We will drop the loan_status column because we already created a
loan_repaid column which is already 0 and 1 """)
df = df.drop('loan_status', axis=1)

st.markdown("Set X and y variables to the .values of the features and label.")
X = df.drop('loan_repaid', axis=1).values
y = df['loan_repaid'].values
st.markdown("Perform a train_test_split")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=101)

st.subheader("Normalizing the Data")

st.markdown("""Use a MinMaxScaler to normalize the feature data X_train
 and X_test. We don't want data leakge from the test set so we only fit
 on the X_train data.""")

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

st.markdown("""Build a Sequential model on which we will train the data
. The model will go 78 --> 39 --> 19 --> 1 output neuron. """)

st.subheader("Creating the Model")
model = Sequential()
# input layer
model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))

# output layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam')

st.markdown("Next, let's fit the model on the training data and save it.")

model.fit(x=X_train,
          y=y_train,
          epochs=25,
          batch_size=256,
          validation_data=(X_test, y_test))

model.save('keras_data_project_model.h5')

st.header("Section 3: Evaluating Model Performance.")
st.markdown("""Time to evaluate how the model performed. Firstly plot out the
 validation loss versus the training loss.""")

losses = pd.DataFrame(model.history.history)
fig12, ax12 = plt.subplots()
losses[['loss', 'val_loss']].plot(ax=ax12)
st.pyplot(fig12)

st.markdown("""Create predictions from the X_test set and display a
 classification report and confusion matrix for the X_test set. """)

predictions = model.predict_classes(X_test)

st.markdown("Classification report")
st.text((classification_report(y_test, predictions)))
st.markdown("Confusion matrix")
st.text(confusion_matrix(y_test, predictions))

st.markdown("Given a customer below, would you offer this person a loan?")

random.seed(101)
random_ind = random.randint(0, len(df))

new_customer = df.drop('loan_repaid', axis=1).iloc[random_ind]
st.text(new_customer)

st.text(model.predict_classes(new_customer.values.reshape(1, 78)))

st.markdown("Did this person actually paid back their loan?")
st.text(df.iloc[random_ind]['loan_repaid'])
st.markdown("""From the result above, the person actually ended up paying back
 the loan as our model predicted. """)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.model_selection import cross_val_score,StratifiedKFold,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,precision_score,f1_score,recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

plt.style.use('seaborn-v0_8')

results = pd.read_csv(r'C:/Users/Allan/Desktop/F1 prdedition/datasets/results.csv')
races = pd.read_csv(r'C:/Users/Allan/Desktop/F1 prdedition/datasets/races.csv')
quali = pd.read_csv(r'C:/Users/Allan/Desktop/F1 prdedition/datasets/qualifying.csv')
drivers = pd.read_csv(r'C:/Users/Allan/Desktop/F1 prdedition/datasets/drivers.csv')
constructors = pd.read_csv(r'C:/Users/Allan/Desktop/F1 prdedition/datasets/constructors.csv')
circuit = pd.read_csv(r'C:/Users/Allan/Desktop/F1 prdedition/datasets/circuits.csv')


df5 = (
    races.merge(results, how='inner', on='raceId')
         .merge(quali, how='inner', on=['raceId', 'driverId', 'constructorId'])
         .merge(drivers, how='inner', on='driverId')
         .merge(constructors, how='inner', on='constructorId')
         .merge(circuit, how='inner', on='circuitId', suffixes=('_left', '_right'))
)

data = df5.drop(columns=['round', 'circuitId', 'time_x', 'url_x', 'resultId', 'driverId',
                         'constructorId', 'number_x', 'positionText', 'position_x',
                         'positionOrder', 'laps', 'time_y', 'rank',
                         'fastestLapTime', 'qualifyId', 'driverRef', 'number', 'code', 'url_y', 'circuitRef',
                         'location', 'lat', 'lng', 'alt', 'number_y', 'points', 'constructorRef', 'name_x', 'raceId', 'fastestLap', 'q2', 'q3', 'milliseconds', 'q1'])


data = data[data['year']>=2010]


data.rename(columns={
    'name': 'GP_name',
    'position_y': 'position',
    'grid': 'quali_pos',
    'name_y': 'constructor',
    'nationality_x': 'driver_nationality',
    'nationality_y': 'constructor_nationality'
}, inplace=True)

data['driver'] = data['forename'] + ' ' + data['surname']
data['date'] = pd.to_datetime(data['date'])
data['dob'] = pd.to_datetime(data['dob'])


data['age_at_gp_in_days'] = abs(data['dob']-data['date'])
data['age_at_gp_in_days'] = data['age_at_gp_in_days'].apply(lambda x: str(x).split(' ')[0])

#Some of the constructors changed their name over the year so replacing old names with current name
data['constructor'] = data['constructor'].apply(lambda x: 'Racing Point' if x=='Force India' else x)
data['constructor'] = data['constructor'].apply(lambda x: 'Alfa Romeo' if x=='Sauber' else x)
data['constructor'] = data['constructor'].apply(lambda x: 'Renault' if x=='Lotus F1' else x)
data['constructor'] = data['constructor'].apply(lambda x: 'AlphaTauri' if x=='Toro Rosso' else x)

data['driver_nationality'] = data['driver_nationality'].apply(lambda x: str(x)[:3])
data['constructor_nationality'] = data['constructor_nationality'].apply(lambda x: str(x)[:3])
data['country'] = data['country'].apply(lambda x: 'Bri' if x=='UK' else x)
data['country'] = data['country'].apply(lambda x: 'Ame' if x=='USA' else x)
data['country'] = data['country'].apply(lambda x: 'Fre' if x=='Fra' else x)
data['country'] = data['country'].apply(lambda x: str(x)[:3])
data['driver_home'] = data['driver_nationality'] == data['country']
data['constructor_home'] = data['constructor_nationality'] == data['country']
data['driver_home'] = data['driver_home'].apply(lambda x: int(x))
data['constructor_home'] = data['constructor_home'].apply(lambda x: int(x))

#reasons for DNF(did not finish)
data['driver_dnf'] = data['statusId'].apply(lambda x: 1 if x in [3,4,20,29,31,41,68,73,81,97,82,104,107,130,137] else 0)
data['constructor_dnf'] = data['statusId'].apply(lambda x: 1 if x not in   [3,4,20,29,31,41,68,73,81,97,82,104,107,130,137,1] else 0)
print(data.columns)

font = {
    'family':'serif',
    'color':'black',
    'weight':'bold',
    'size':10
}

dnf_by_driver = data.groupby('driver')['driver_dnf'].sum(numeric_only=True)
driver_race_entered = data.groupby('driver').count()['driver_dnf']
driver_dnf_ratio = (dnf_by_driver/driver_race_entered)
driver_confidence = 1-driver_dnf_ratio
driver_confidence_dict = dict(zip(driver_confidence.index,driver_confidence))

dnf_by_constructor = data.groupby('constructor')['constructor_dnf'].sum(numeric_only=True)

constructor_race_entered = data.groupby('constructor').count()['constructor_dnf']
constructor_dnf_ratio = (dnf_by_constructor/constructor_race_entered)
constructor_relaiblity = 1-constructor_dnf_ratio
constructor_relaiblity_dict = dict(zip(constructor_relaiblity.index,constructor_relaiblity))

data['driver_confidence'] = data['driver'].apply(lambda x:driver_confidence_dict[x])
data['constructor_relaiblity'] = data['constructor'].apply(lambda x:constructor_relaiblity_dict[x])
#removing retired drivers and constructors
active_constructors = ['Renault', 'Williams', 'McLaren', 'Ferrari', 'Mercedes',
                       'AlphaTauri', 'Racing Point', 'Alfa Romeo', 'Red Bull',
                       'Haas F1 Team']
active_drivers = ['Daniel Ricciardo', 'Kevin Magnussen', 'Carlos Sainz',
                  'Valtteri Bottas', 'Lance Stroll', 'George Russell',
                  'Lando Norris', 'Sebastian Vettel', 'Kimi Räikkönen',
                  'Charles Leclerc', 'Lewis Hamilton', 'Daniil Kvyat',
                  'Max Verstappen', 'Pierre Gasly', 'Alexander Albon',
                  'Sergio Pérez', 'Esteban Ocon', 'Antonio Giovinazzi',
                  'Romain Grosjean','Nicholas Latifi']
data['active_driver'] = data['driver'].apply(lambda x: int(x in active_drivers))
data['active_constructor'] = data['constructor'].apply(lambda x: int(x in active_constructors))

cleaned_data = data[['GP_name','quali_pos','constructor','driver','position','driver_confidence','constructor_relaiblity','active_driver','active_constructor','dob']]
cleaned_data = cleaned_data[(cleaned_data['active_driver']==1)&(cleaned_data['active_constructor']==1)]
cleaned_data.to_csv('cleaned_data.csv',index=False)
print(cleaned_data.columns)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def position_index(x):
    if x<4:
        return 1
    
    if x>10:
        return 3
    else :
        return 2
    
print(cleaned_data.columns)


sc = StandardScaler()
le = LabelEncoder()

cleaned_data['GP_name'] = le.fit_transform(cleaned_data['GP_name'])
cleaned_data['constructor'] = le.fit_transform(cleaned_data['constructor'])
cleaned_data['driver'] = le.fit_transform(cleaned_data['driver'])
X = cleaned_data.drop(['position', 'active_driver', 'active_constructor'], axis=1)
y = cleaned_data['position'].apply(lambda cleaned_data: position_index(cleaned_data))  # Assuming position_index is defined elsewhere


X.drop('constructor_relaiblity', axis=1, inplace=True)

X.drop('driver_confidence', axis=1, inplace=True)

X.drop('dob', axis=1, inplace=True)

print("Data Types of X:")
print(X.dtypes)

print("\nData Type of y:")
print(y.dtypes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = [ RandomForestClassifier()]
names = ['RandomForestClassifier']
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Assuming 'X' is your features and 'y' is your labels
model = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', C=0.01, solver='liblinear',max_iter=1000))

model_dict = dict(zip(models, names))
mean_results = []
results = []
name = []
model = LogisticRegression(penalty='l1', C=0.01, solver='liblinear',max_iter=1000)

model.fit(X, y)
pickle.dump(model,open('trial.pkl','wb'))
model=pickle.load(open('trial.pkl','rb'))
# for model in models:
#     cv = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)  # Added shuffle=True for better randomness
#     result = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
#     print(f'{model_dict[model]} : {result.mean()}')
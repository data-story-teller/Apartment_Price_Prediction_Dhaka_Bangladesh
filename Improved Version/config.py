PATH_TO_DATASET = "dhaka_city_apartment_price.csv"
OUTPUT_MODEL_PATH = "regression_model.pkl"
OUTPUT_SCALER_PATH = "scaler.pkl"

# set parameters

NUMERICAL_VARS = ['bed','bath','total_sqft','price']
DISCRETE_VARS = ['bed', 'bath']
CATEGORICAL_VARS = ['zone', 'area']
CONT_VARS = ['total_sqft', 'price']


# encoding parameters
FREQUENT_LABELS = {
    'area': ['Adabor', 'Aftab Nagar', 'Agargaon', 'Badda', 'Banasree', 'Bashabo',
       'Bashundhara R-A', 'Dakshin Khan', 'Dhanmondi', 'Gulshan', 'Hazaribag',
       'Khilgaon', 'Malibagh', 'Mirpur', 'Mohammadpur', 'Mugdapara', 'Rampura',
       'Savar', 'Uttara'],

    'zone': ['Ashkona', 'Block A', 'Block B', 'Block C', 'Block D', 'Block E',
       'Block F', 'Block G', 'Block I', 'Chandrima Model Town', 'East Rampura',
       'Faydabad', 'Pallabi', 'Section 1', 'Section 10', 'Section 12',
       'Section 2', 'Sector 10', 'Shahjadpur', 'South Banasree Project',
       'Uttar Badda', 'West Dhanmondi and Shangkar', 'West Rampura']}

ENCODING_MAPPINGS = {'area': {'Savar': 0, 'Dakshin Khan': 1, 'Khilgaon': 2, 'Mirpur': 3, 'Hazaribag': 4, 'Agargaon': 5, 'Mohammadpur': 6, 'Bashabo': 7,
                              'Mugdapara': 8, 'Rampura': 9, 'Badda': 10, 'Banasree': 11, 'Adabor': 12, 'Malibagh': 13, 'Rare': 14, 'Aftab Nagar': 15,
                              'Dhanmondi': 16, 'Uttara': 17, 'Bashundhara R-A': 18, 'Gulshan': 19},

                     'zone': {'Faydabad': 0, 'Section 12': 1, 'Chandrima Model Town': 2, 'Section 1': 3, 'East Rampura': 4, 'Uttar Badda': 5,
                             'Section 10': 6, 'Section 2': 7, 'South Banasree Project': 8, 'Ashkona': 9, 'Pallabi': 10, 'West Rampura': 11, 'Rare': 12,
                             'West Dhanmondi and Shangkar': 13, 'Sector 10': 14, 'Block E': 15, 'Block F': 16, 'Block G': 17, 'Shahjadpur': 18,
                             'Block B': 19, 'Block I': 20, 'Block D': 21, 'Block C': 22, 'Block A': 23}
                     }

# ======= FEATURE GROUPS =============
# variable groups for engineering steps
TARGET = 'price'

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV


#
# Model pipeline
#

# We create the preprocessing pipelines for both numeric and categorical data.
#numeric_features = ["if"+str(i) for i in range(1,14)]
#numeric_transformer = Pipeline(steps=[
#    ('imputer', SimpleImputer(strategy='median')),
#    ('scaler', StandardScaler())
#])

#categorical_features = ["cf"+str(i) for i in range(1,27)] + ["day_number"]
#categorical_transformer = Pipeline(steps=[
#    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
#    ('onehot', OneHotEncoder(handle_unknown='ignore'))
#])


#
# Dataset fields
#
#fields = ["id"] + numeric_features + categorical_features
#fields_t = ["id","label"] + numeric_features + categorical_features



numeric_features = ["if"+str(i) for i in range(1,14)]
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ["cf1","cf2","cf3","cf5","cf6","cf7","cf10","cf11","cf12","cf13","cf20","cf21","cf22","cf23","cf24"]
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])



preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Now we have a full prediction pipeline.
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('linearregression', LinearRegression())
])

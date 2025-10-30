import numpy as np

chestpain_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
restingrelectro_map = {"Normal": 0, "ST-T Abnormality": 1, "Left Ventricular Hypertrophy": 2}
slope_map = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
noofmajorvessels_map = {"0": 0, "1": 1, "2": 2, "3": 3}

numeric_cols = ['age', 'restingBP', 'maxheartrate', 'oldpeak', 'serumcholestrol']
cat_cols = ['chestpain', 'restingrelectro', 'slope', 'noofmajorvessels']
binary_cols = ['gender', 'fastingbloodsugar', 'exerciseangia']

def preprocess_input(df, scaler, encoder):
    # Map categorical columns
    df['chestpain'] = df['chestpain'].map(chestpain_map)
    df['restingrelectro'] = df['restingrelectro'].map(restingrelectro_map)
    df['slope'] = df['slope'].map(slope_map)
    df['noofmajorvessels'] = df['noofmajorvessels'].map(noofmajorvessels_map)

    scaled_numeric = scaler.transform(df[numeric_cols])
    encoded_cat = encoder.transform(df[cat_cols])
    binary_features = df[binary_cols].to_numpy()

    features = np.hstack([scaled_numeric, encoded_cat, binary_features])
    return features

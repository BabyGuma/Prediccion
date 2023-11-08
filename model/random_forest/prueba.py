import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from joblib import load
from sklearn.preprocessing import LabelEncoder

model = load('spotify.joblib')
le_y = load('le_y.joblib')
le_X = LabelEncoder()


date = datetime(2023, 1, 1)

end_date = datetime(2024, 1, 1)

data = []

while date < end_date:
    data.append({
        'year': date.year,
        'month': date.month,
        'day': date.day,
        'hour': date.hour
    })
    date += timedelta(hours=1)

df = pd.DataFrame(data)
predictions = model.predict(df)
print(le_y.inverse_transform(predictions))

sunny_inputs = df[le_y.inverse_transform(predictions) == "['Taylor Swift']"]
print("Taylor inputs:")
print(sunny_inputs)

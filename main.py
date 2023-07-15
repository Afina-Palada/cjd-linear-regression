import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

file1_data = pd.read_csv('СВОД 2022.csv')
file2_data = pd.read_csv('СВОД 2023 6 мес.csv')
data = pd.concat([file1_data, file2_data], ignore_index=True).drop(['сумма по билетам', 'сумма по плацкарту'], axis=1)
data = pd.get_dummies(data, columns=['поезд', 'станция отправления', 'станция назначения', 'тип'])

X = data.drop(['итоговая сумма билета'], axis=1)
y = data['итоговая сумма билета']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_normalized, y_train)

mse = round(model.score(scaler.transform(X_test), y_test) * 100)
print(f"Точность: {mse}%")

new_data = pd.DataFrame({
    'поезд': ['0306С/А'],
    'месяц поездки': [1],
    'месяц': [1],
    'станция отправления': ['АДЛЕР'],
    'станция назначения': ['МОСКВА КАЗ'],
    'тип': ['П/3Б'],
    'количество мест': [1],
    'сумма сервисных услуг': [163.4],
    'расстояние': [1790]
})
new_data_encoded = pd.get_dummies(new_data, columns=['поезд', 'станция отправления', 'станция назначения', 'тип'])
new_data_encoded = new_data_encoded.reindex(columns=X.columns, fill_value=0)
new_data_normalized = scaler.transform(new_data_encoded)

predicted_price = round(model.predict(new_data_normalized)[0], 1)
print(f"Предсказанная цена билета: {predicted_price}")

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

files = []
file_names = ['data.csv', 'data2.csv', 'data3.csv', 'data4.csv', 'data5.csv']
for file_name in file_names:
    files.append(pd.read_csv(file_name))

data = pd.concat(files, ignore_index=True).drop([
    'Сумма по бил.',
    'Сумма по плац.',
    'Поезд/Нитка',
    'Станция отп.',
    'Станция назн.'
], axis=1)
data = pd.get_dummies(data, columns=['Тип/Кл.обсл.'])

X = data.drop('Итого сумма билета', axis=1)
y = data['Итого сумма билета']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_normalized, y_train)

mse = round(model.score(scaler.transform(X_test), y_test) * 100)
print(f"Точность: {mse}%")

new_data = pd.DataFrame({
    'Дней до отправления': [5],
    'Тип/Кл.обсл.': ['3Э'],
    'Кол-во прод. мест': [1],
    'Сумма серв. усл.': [400],
    'Расстояние': [2500]
})

new_data_encoded = pd.get_dummies(new_data, columns=['Тип/Кл.обсл.'])
new_data_encoded = new_data_encoded.reindex(columns=X.columns, fill_value=0)
new_data_normalized = scaler.transform(new_data_encoded)

predicted_price = round(model.predict(new_data_normalized)[0], 1)
print(f"Предсказанная цена билета: {predicted_price}")

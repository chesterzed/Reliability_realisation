from matplotlib import pyplot as plt
import pandas as pd

# считывание датасета
data = pd.read_csv('MotorA.csv')
# Убрал пробел в названии столбца
data.columns = [data.columns[0], 'Censoring_Indicator', data.columns[2]]
# Выбрал только сроки с отказами
t = data[data['Censoring_Indicator'] == 'Failed']
# Вывел на экран
print(t)

# А далее, составил список с временами отказа,
# умножая период, на количество отказов в нём
f = []
g = list(t['Days'])
h = list(t['Count'])
for i in range(len(g)):
    f.extend([g[i]] * h[i])
print(f)


fig, axs = plt.subplots(nrows=1, ncols=2)
axs = axs.ravel()
axs[0].scatter(list(t['Days']), list(t['Count']))
axs[1].hist(f)
plt.show()


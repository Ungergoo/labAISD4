# Формируется матрица F следующим образом: скопировать в нее А и если в С
# количество четных чисел в нечетных столбцах больше, чем сумма чисел в нечетных
# строках, то поменять местами С и Е симметрично, иначе В и Е поменять местами
# несимметрично. При этом матрица А не меняется. После чего если определитель
# матрицы А больше суммы диагональных элементов матрицы F, то вычисляется
# выражение: A*AT – K * F-1, иначе вычисляется выражение (AТ +G-FТ)*K, где G-нижняя
# треугольная матрица, полученная из А. Выводятся по мере формирования А, F и все
# матричные операции последовательно.

import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np


k = int(input("Введите число K являющееся коэффициентом при умножении: "))
n = int(input("Введите число n больше 3 которое являеться рзмером матрицы: "))
while n <= 3:
    n = int(input("\nВведите число больше 3 "))
A = np.random.randint(-10, 10, (n, n))
# A = np.ones((n,n))      # задание единичной матрицы
print("Матрица A:\n", A)


half_n = n // 2
maxfix_n = half_n
minfix_n = half_n
if n % 2 != 0:
    maxfix_n += 1
    minfix_n = maxfix_n - 1


F = A.copy()
B = np.array(A[:minfix_n, :minfix_n])
print('Подматрица B матрицы A:\n', B)
C = np.array(A[:minfix_n, maxfix_n:])
print('Подматрица C матрицы A:\n', C)
E = np.array(A[maxfix_n:, maxfix_n:])
print('Подматрица E матрицы A:\n', E)
D = np.array(A[maxfix_n:, :minfix_n])
print('Подматрица D матрицы A:\n', D)


odd_C = (C[:, 1::2] % 2 == 0).sum()
oddnt_C = (C[1::2, :]).sum()

if oddnt_C < odd_C:
    print("Меняем симметрично C и E")
    F[:minfix_n, :minfix_n] = C[-1::-1, :minfix_n]
    F[maxfix_n:, maxfix_n:] = B[-1::-1, :minfix_n]
else:
    print("Меняем несимметрично B и Е")
    F[:minfix_n, maxfix_n:] = E
    F[maxfix_n:, maxfix_n:] = B

print(F)
tA = np.transpose(A)
tF = np.transpose(F)
det_A = np.linalg.det(A)
diag_F = np.trace(F)

if det_A > diag_F:
    print('Вычисляем выражение :  A * AT – K * F-1 ')

    mA = np.dot(A, tA)
    print('Умножение A * AT\n', mA)

    pF = np.linalg.matrix_power(F, -1)
    print('Возведение матрицы F в -1 степень:\n', pF)

    KF = np.dot(k, pF)
    print('Умножение K * F-1 \n', KF)

    result = np.subtract(mA, KF)
    print('Разница матриц\n', result)
else:
    print('Вычисляем выражение:((AТ + G - FТ ) * K ')
    print('Транспонированая матрица A:\n', tA)

    print('Транспонированая матрица F:\n', tF)
    G = np.tril(A)
    print('Нижняя треугольная матрица G из матрицы A:\n', G)

    ptAG = np.add(tA, G)
    print('Сумма AT + G:\n', ptAG)

    pAGFT = np.subtract(ptAG, tF)
    print(*'Разница AT + G - FT:\n', pAGFT)

    result = np.dot(pAGFT, k)
    print(*'Умножение на K\n', result)

print('Результат вычислений\n', result)


explode = [0] * (n - 1)
explode.append(0.1)
plt.title("Круговая диаграмма")
try:
    sizes = [round(np.mean(abs(F[i, ::])) * 100, 1) for i in range(n)]
except IndexError:
    sizes = [round(np.mean(abs(F[i, ::])) * 100, 1) for i in range(n)]
plt.pie(sizes, labels=list(range(1, n + 1)), explode=explode, autopct='%1.1f%%', shadow=True)
plt.show()

plt.plot(A)
plt.title("График")
plt.ylabel("y axis")
plt.xlabel("x axis")
plt.show()

sns.heatmap(A, cmap="Spectral", annot=True)
plt.title("Тепловая карта")
plt.ylabel("Номер строки")
plt.xlabel("Номер столбца")
plt.show()
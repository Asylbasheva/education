import numpy as np

def act(x):
    return 0 if x < 0.5 else 1

def go(color, size, design):
    x = np.array([color, size, design])
    w11 = [0.3, 0.3, 0]
    w12 = [0.4, -0.5, 1]
    weight1 = np.array([w11, w12])
    weight2 = np.array([-1, 1])

    sum_hidden = np.dot(weight1, x)
    print("Значение суммы на нейроннах скрытого слоя: " +str(sum_hidden))

    out_hidden = np.array([act(x) for x in sum_hidden])
    print("Значение на выходах нейроннов скрытого слоя: " +str(out_hidden))

    sum_end = np.dot(weight2, out_hidden)
    y = act(sum_end)
    print("выходное значение НС: "+str(y))

    return y

color = 0
size = 1
design = 1

res = go(color, size, design)
if res == 1:
    print("Покупаем")
else:
    print("Ищем другое")






import numpy as np


class GoldenSectionSearch:
    def search(self, f, a_init, b_init, epsilon):
        c = (3 - np.sqrt(5)) / 2
        a, b, y, z, f_y, f_z = [], [], [], [], [], []
        y.append(a_init + c * (b_init - a_init))
        z.append(a_init + (1 - c) * (b_init - a_init))
        f_y.append(f(y[-1]))
        f_z.append(f(z[-1]))

        if f_y[-1] <= f_z[-1]:
            b.append(z[-1])
            a.append(a_init)
        else:
            a.append(y[-1])
            b.append(b_init)

        itr = 0
        while b[-1] - a[-1] > epsilon:
            itr += 1
            if f_y[-1] <= f_z[-1]:
                z.append(y[-1])
                f_z.append(f_y[-1])
                y.append(a[-1] + c * (b[-1] - a[-1]))
                f_y.append(f(y[-1]))
            else:
                y.append(z[-1])
                f_y.append(f_z[-1])
                z.append(a[-1] + (1 - c) * (b[-1] - a[-1]))
                f_z.append(f(z[-1]))

            if f_y[-1] <= f_z[-1]:
                a.append(a[-1])
                b.append(z[-1])
            else:
                a.append(y[-1])
                b.append(b[-1])

        if f_y[-1] <= f_z[-1]:
            x = y[-1]
            f_x = f_y[-1]
        else:
            x = z[-1]
            f_x = f_z[-1]

        return x, f_x

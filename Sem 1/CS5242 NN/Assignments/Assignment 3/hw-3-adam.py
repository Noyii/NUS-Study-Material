# Function
# f = (1-x), x <= 1
# f = (x-1), 1 < x <= (h+1)
# f = (1-x), (h+1) < x <= (1+2h)
# f = c(1-x), x > (1+2h)

# Derivative
# f` = -1, x <= 1
# f` = 1, 1 < x <= (h+1)
# f` = -1, (h+1) < x <= (1+2h)
# f` = -c, x > (1+2h)


def adam(a, b1, b2):
    x = 0 # Initial value at point O
    x_prev = 0
    m = 0
    v = 0
    t = 0
    # x_vals = [x_prev]

    while True: 
        t += 1
        g = get_gradient(x)
        # print(g)

        m = b1*m + (1 - b1)*g
        v = b2*v + (1 - b2)*g*g

        m_norm = m/(1 - b1**t)
        v_norm = v/(1 - b2**t)

        x_prev = x
        x = x_prev - a * m_norm/(v_norm**0.5) 
        # x_vals.append(x)

        if (x < x_prev):
            # it has started to go towards the local minima
            break

    return x_prev
    # return x_prev, x_vals

def get_gradient(x):
    # Only adding gradients for the first 2 peicewise linear functions
    # since we only need to escape local minima
    if x <= 1:
        return -1
    if x > 1:
        return 1

def main():
    # a = 0.3
    a = 0.001
    b1 = 0.9
    b2 = 0.999

    # x, x_vals = adam(a, b1, b2)
    x_max = adam(a, b1, b2)
    h_max = x_max - 1

    # print(x_vals)
    # print(x_max)
    print("Max height h of the bump in which the Adam optimizer will escape the local min at x = {}".format(h_max))

if __name__ == "__main__":
    main()

import sys
def check_hailstone(x, a, b):
    items = [x]
    max_int = int(3000)
    while True:
        if x % 2 == 0:
            x = int(x / 2)
        else:
            x = a * x + b
        if x in items and x % 2 != 0:
            items.append(x)
            return True, items
        elif x > max_int:
            return False, []
        
        items.append(x)

        


for x in range(1, 11):
    for a in range(1, 11):
        for b in range(1, 11):
            is_hailstone, items = check_hailstone(x, a, b)
            if is_hailstone:
                second_last_index = items[:-1].index(items[-1])
                print('x = {}, a = {}, b = {}, seq = {}'.format(x, a, b, items[second_last_index:]))
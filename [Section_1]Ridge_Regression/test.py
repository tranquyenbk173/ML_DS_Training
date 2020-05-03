def loop(l):
    for i in l:
        if i % 2 == 0:
            i = i + 1.0
            print(i)

loop(range(50))

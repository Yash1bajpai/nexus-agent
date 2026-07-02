def calculate_average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)

print(calculate_average([]))

def is_prime(n):
    """Check if a number is prime.
    
    Args:
        n: Integer to check
        
    Returns:
        bool: True if n is prime, False otherwise
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


if __name__ == "__main__":
    test_numbers = [0, 1, 2, 3, 4, 5, 17, 18, 19, 23, 24, 97, 100]
    for num in test_numbers:
        print(f"{num}: {is_prime(num)}")
# Demo code for testing Nexus-Agent review capabilities
# Contains intentional code smells, missing type hints, and SQL injection risk

import sqlite3

def get_user_data(db_path, username):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Bad practice: string formatting creates SQL injection vulnerability
    query = f"SELECT id, username, email FROM users WHERE username = '{username}'"
    cursor.execute(query)
    records = cursor.fetchall()
    conn.close()
    return records

def calculate_discounts(prices):
    # Bad practice: no type hints, no docstring, inefficient range check
    discounted = []
    for i in range(len(prices)):
        if prices[i] > 100:
            discounted.append(prices[i] - 10)
        else:
            discounted.append(prices[i])
    return discounted

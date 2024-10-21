import pandas as pd

def fibonacci(n):
    """Generate a list of the first n Fibonacci numbers."""
    fib_sequence = []
    a, b = 0, 1
    for _ in range(n):
        fib_sequence.append(a)
        a, b = b, a + b
    return fib_sequence

def create_fibonacci_dataframe(n):
    """Create a DataFrame containing the first n Fibonacci numbers."""
    fib_numbers = fibonacci(n)
    df = pd.DataFrame({
        'Fibonacci Number': fib_numbers
    })
    return df

# Number of Fibonacci numbers to generate
num_fib_numbers = 10

# Create the DataFrame
df = create_fibonacci_dataframe(num_fib_numbers)

# Display the DataFrame
print(df)

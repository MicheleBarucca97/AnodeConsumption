import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x)
def f(x):
    return x**3 - x - 3

# Define the derivative of the function f'(x)
def f_prime(x):
    return 3*x**2 - 1

def f_prime_approximated(x, xold):
    return (f(x) - f(xold)) / (x -xold)

# Implement Newton's method with convergence tracking
def newton_method(f, f_prime, initial_guess, tolerance=1e-7, max_iterations=1000):
    x = initial_guess
    x_values = [x]  # List to store x values for plotting
    for i in range(max_iterations):
        fx = f(x)
        if i <= 0:
            fpx = f_prime(x)
        else:
            fpx = f_prime_approximated(x, xold)
        if fpx == 0:
            print("Zero derivative. No solution found.")
            return None, x_values
        x_new = x - fx / fpx
        x_values.append(x_new)
        if abs(x_new - x) < tolerance:
            return x_new, x_values
        xold = x
        x = x_new

    print("Exceeded maximum iterations. No solution found.")
    return None, x_values

# Parameters for Newton's method
initial_guess = 0.0
tolerance = 1e-7
max_iterations = 1000

# Find the root using Newton's method
root, x_values = newton_method(f, f_prime, initial_guess, tolerance, max_iterations)

# Plot the function f(x)
x = np.linspace(-2, 3, 400)
y = f(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='$f(x) = x^3 - x - 3$', color='blue')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# Plot Newton iterates and annotate with iteration numbers
for i, xi in enumerate(x_values):
    plt.scatter(xi, f(xi), color='red', zorder=5)
    plt.text(xi, f(xi), f'{i}', fontsize=9, ha='right')

plt.plot(x_values, [f(xi) for xi in x_values], color='red', linestyle='--', linewidth=0.5, label='Newton iterates')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Newton\'s Method Convergence')
plt.legend()
plt.grid(True)
plt.show()

if root is not None:
    print(f"The root found is: {root}")
    print(f"f({root}) = {f(root)}")
else:
    print("No root found.")

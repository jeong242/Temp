import math

# Define some constants.
sqrt5 = math.sqrt(5)
phi = (1+sqrt5)/2

def fib(n):
	return math.floor(math.pow(phi,n+1)/sqrt5 + 0.5)

# Return list of fib sequence up to nth.
def fib_list(n):
	list = []
	for i in range(n):
		list += [fib(i)]
	return list

#############
# Lists
#############

l = ['first', 'second', 'third']        # Creating a list
l.append('fourth')                      # Adding to a list
item1 = l[1]                            # Item in position 1
item2 = l[-1]                           # Item in the last position
l[1] = 'changed'                        # Updating an item if you know the position
my_index = l.index('third')             # Finding the position of an element in a list
check_result = 'sixth' in l             # Checking to see if an item is in a list
l. insert(1, 'Rhubarb')  # Inserting an item into a list
slice2 = l[:2]                          # Slice from beginning of the list to position 2 exclusive
slice1 = l[0:2]                         # Slice from position 0 to 2 exclusive
slice3 = l[1:3]                         # Slice from the middle
for i in l: print(i)                    # Iterating over a list
length = len(l)                  # Length of a list
del l[1]                                # Removing an item where you know the index
l.remove('first')                       # Removing an item where you know the value
copy_list = l[:]                        # Using slices to copy a list
# Enumerate tracks the index of each item as you loop through the list
for index, item in enumerate(l):
    place = str(index+1)
    print('Place ' + place + ' Item ' + item.title())
# Methods = Something that can be done to a variable
l.sort(reverse=True)
# Functions = Called with parameters
new_list = sorted(l, reverse=True)
# Working with a numerical list
ages = [10, 20, 46, 78, 90, 12, 17, 18]
print(min(ages))
print(max(ages))
print(sum(ages))

# Strings are lists of characters
message = 'Hello Patrick Cresswell'
message_list = list(message)
print(message_list)
first_char = message[0]
print(first_char)
check_present = 'Pat' in message
print(check_present)
pat_index = message.find('Pat')
print(pat_index)
message = message.replace('Patrick', 'Sheena')
print(message)
num_times = message.count('She')
words = message.split(' ')
print(words)

# String formatting using placeholders
animals = ('dog', 'cat', 'bear')
print('I have a %s and a %s and a %s' % (animals[0], animals[1], animals[2]))
answer = 42
print('Answer to life the universe and everything is %d' % answer)

# Tuples : Just like lists but once created cannot change them
# Why would use ? More memory efficient
tuple1 = (50, 'hello', 'dog')
print(tuple1[1])

# Sets : Can be added to. No duplicates. No ordering
X = {1, 2, 3, 3, 4}
X.add(5)
Y = {1, 2, 3}
Z1 = X.union(Y)
print(Z1)
Z2 = X.intersection(Y)
print(Z2)

# Dictionaries : Can be added to and updated
students = { 100: 'Bob', 101: 'Jim', 102: 'Sue'}
print(students[101])
students[100] = 'Tom'
students[103] = 'Pat'
print(students)

# Loops

for i in [0, 1, 2, 3]:
    print('Hello')
# Creating an empty dictionary then populating using iteration
result = dict()
for i in range(1, 10, 2):       # Range - taking a start number, end number and step number
    result[i] = i**2
print(result)

# Floating point numbers : Has a problem when can't represent internally as a power of 2
result = 0.1 + 0.2
print(result)

# Logical tests
a = 1
b = 2
if a == b:
    print('Equal')
elif a > b:
    print('a is more than b')
else:
    print('a is less than b')

# Functions : Parameters are local - this has no impact
def assign(x, y):
    x = y
x = 0
assign (x, 5)
print(x)
# Functions : Mutable data eg lists are passed by reference - so CAN be altered
def update_list(lst):
    lst.append('z')
my_list = ['a', 'b', 'c']
update_list(my_list)
print(my_list)

# Map function
# Applies a function to each element of a collection eg a list
def squared(x):
    result = x ** 2
    return result
result = list(map(squared, [1, 2, 3, 4]))
print(result)
# Same functionality as list comprehension
same_result = [squared(i) for i in [1, 2, 3, 4]]
print(same_result)
# Same functionality using a lambda functions
# Anonymous function literal
same_again = list(map(lambda x: x ** 2, [1, 2, 3, 4]))
print(same_again)

# Reduce function
# Recursively applies another function to pairs of values over the list
# Resulting in a SINGLE return value
from functools import reduce
result = reduce(lambda x, y: x + y, [0, 1, 2, 3, 4, 5])
print(result)

# MapReduce is a programming model for processing and generating big data sets
# with a parallel, distributed algorithm on a cluster
# A MapReduce program is composed of a map procedure, which performs filtering
# and sorting and a reduce method, which performs a summary operation

# Filtering data
result = list(filter(lambda x: x != ' ', 'hello world'))
print(result)
# Same functionality as list comprehension
same_result = [x for x in 'hello world' if x != ' ']
print(same_result)

# Cartesian product using list comprehension
A = ['x', 'y', 'z']
B = [1, 2, 3]
result = [(a, b) for a in A for b in B]
print(result)

# While loops : Processing a list
unconfirmed_users = ['bob', 'tom', 'pat', 'she']
confirmed_users = []
while len(unconfirmed_users) > 0:
    current_user = unconfirmed_users.pop()
    print('Confirming ' + current_user)
    confirmed_users.append(current_user)
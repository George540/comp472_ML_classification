##########################################
## main.py (Mini-Assignment 1 COMP 472)
## This code contains the execution of the program's main menu
## Created by Team Oranges
##########################################

print('Welcome to Team Oranges Mini-Assignment 1 for COMP 472')
option = int(input('Please select one of the following options:\n1 - Run Task 1\n2 - Run Task 2\n3 - Run Task 1 and Task 2 back-to-back\n'))
if (option == 1):
    print('Initializing Task 1, this may take a few seconds...\n\n')
    import taskOne
    taskOne.runTaskOne()
    print('--------------Execution of Task 1 Complete!--------------')
    print('BBC-distribution.pdf has been updated...')
    print('bbc-performance.txt has been updated...')
elif (option == 2):
    print('Initializing Task 2, this may take a few seconds...')
    import taskTwo
    taskTwo.runTaskTwo()
    print('--------------Execution of Task 2 Complete!--------------')
    print('drug-distribution.pdf has been updated...')
    print('drug-performance.txt has been updated...')
elif (option == 3):
    print('Initializing Task 1 and Task 2, this may take a few seconds...')
    import taskOne
    taskOne.runTaskOne()
    print('--------------Execution of Task 1 Complete!--------------')
    print('BBC-distribution.pdf has been updated...')
    print('bbc-performance.txt has been updated...')
    print('\n\nInitializing Task 2, this may take a few seconds...')
    import taskTwo
    taskTwo.runTaskTwo()
    print('--------------Execution of Task 2 Complete!--------------')
    print('drug-distribution.pdf has been updated...')
    print('drug-performance.txt has been updated...')
else:
    print('Invalid input, system will exit...')
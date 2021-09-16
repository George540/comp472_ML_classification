import os

dir = 'BBC\\business'

with os.scandir(dir) as entries:
    total = 0
    for entry in entries:
        if entry.is_file():
            with open(entry, "r") as file:
                data = file.read()
                occurences = data.count(dir[4:])
                total += occurences
    print(total)
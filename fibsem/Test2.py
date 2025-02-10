import os
def read_from_dict(filename=str):
    """
    The user should create txt files for his experiment conditions which are then converted to dicts and used
    to set the milling/imaging conditions.
    """
    dictionary = {}
    desktop_path = '/Users/sophia.betzler/Desktop/TestImages/20250206'
    with open(os.path.join(desktop_path + '/' + filename),'r') as file:
        for line in file:
            if ":" in line:  # Ensure it's a key-value pair
                key, value = line.strip().split(":", 1)  # Split on first ":"
                dictionary[key.strip()] = value.strip()
    return dictionary



dict = read_from_dict('imaging.txt')
print(dict)
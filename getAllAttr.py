import random

priceList = [5000,742,9938,2129,15393,3497,2010,8858,13400,1868]


dictUsers = {
             'likith': 1000028,
             'sahithi': 1000005
             }


dictPrice = {
             'likith': 8690,
             'sahithi': 12131,
             'not anyone': random.choice(priceList)
            }

cityName = 'A'




def getUserId(name):
    if name.lower() in dictUsers:
        userId = dictUsers.get(name)
        return userId
    else:
        return -1

def getCityName():
    return cityName



def getPriceForName(name):
    if name.lower() in dictPrice:
       return dictPrice.get(name)
    else:
        return random.choice(priceList);

def convertAges(age):
    if age is '(0,2)' or age is '(2,6)' or age is '(6,14)':
        return '(0,14)'
    else:
        return age


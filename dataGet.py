import random

myDict = {
          1: ['cup noodles', 'Maggie', 'chips'],
          2: ['Coke', 'Mountain Dew', 'Red bull'],
          3: ['Yogurt', 'Milk', 'Cheese'],
          4: ['Moxed Veg', 'Pizza', 'Meat'],
          5: ['Beer', 'Wine', 'Vodka'],
          6: ['Sun Screen', 'Lip Balm', 'Deodarant'],
          7: ['Chocolate', 'Cereals', 'Honey'],
          8: ['Napkins', "Tissues", 'Baby powder'],
          9: ['Shampoo', 'Body wash', 'Face Wash'],
          10: ['Sweaters', 'Night wear', 'Coats'],
          11: ['Detergent', 'Carpet','Mop'],
          12: ['Salt', 'Oil', 'Dish Soap'],
          13: ['Hulk Toy', 'Batman Toy', 'Barbie'],
          14: ['Golf kit', 'Running Tracks', 'Baseball'],
          15: ['Glasses', 'Eye Cream', 'Dark circle remover'],
          16: ['saw', 'wood', 'tool kit'],
          17: ['Spinach', 'Tomato', 'Grapes'],
          18: ['Ear phones', 'C Charger','Cable']
          }

def suggestProduct(category):
    prod =[]
    temp = []
    if category[0] in myDict:
        prod = myDict.get(category[0])
        temp =[prod,random.choice(prod)]
        return temp


from random import random
from random import seed
import math
seed()
pi=0
p=0
sq=0
while True:
   x=random()
   y=random()
   t=math.sqrt(x**2+y**2)
   sq=sq+1
   if t<=1:
      p=p+1
   pi=p/sq*4
   if (sq % 100000) ==0:
      print(sq," ",pi,"  ",math.pi-pi)
   

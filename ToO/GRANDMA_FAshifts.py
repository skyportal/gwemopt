#!/usr/bin/env python
import numpy as np
from datetime import datetime

def Calendar():
 West=open("GRANDMA_FA_west.txt").readlines()[0:-1]
 #print(West)
 East=open("GRANDMA_FA_east.txt").readlines()[0:-1]
 len_west=len(West)
 len_east=len(East)
 Shift=[]
 West_list=[]
 East_list=[]

 for l in np.arange(52):
  for m in np.arange(len_west):
    West_list.append(West[m].split("\n")[0])
  for n in np.arange(len_east):
    East_list.append(East[n].split("\n")[0])

 West_list=np.array(West_list) 
 East_list=np.array(East_list)
 
 #for i in np.arange(52):
  #Shift.append(["","","",""])

 #Shift=np.array(Shift)  
 Shift_west=[]
 re=0
 for k in np.arange(52/(len_west/2)+1):
  if k!=52/(len_west/2):
   for h in np.arange(len_west/2):
     Shift_west.append([West_list[h*2],West_list[((h+k)*2+1)]])
  if k==52/(len_west/2):
   for h in np.arange(52%(len_west/2)):
     Shift_west.append([West_list[h*2],West_list[((h+k)*2+1)]])
 #print(len(Shift_west))

 Shift_east=[]
 for k in np.arange(52/(len_east/2)+1):
  if k!=52/(len_east/2):
   for h in np.arange(len_east/2):
     Shift_east.append([East_list[h*2],East_list[((h+k)*2+1)]])
  if k==52/(len_east/2):
   for h in np.arange(52%(len_east/2)):
     Shift_east.append([East_list[h*2],East_list[((h+k)*2+1)]])


 Shift_east=np.array(Shift_east)
 Shift_west=np.array(Shift_west)

 for i in np.arange(52):

  Shift.append([Shift_east[i,0],Shift_east[i,1],Shift_west[i,0],Shift_west[i,1]])
 Shift=np.array(Shift)


 return Shift

def On_duty():
  timenow = str(datetime.now())
  year = int(str(timenow.split()[0]).split("-")[0])
  month = int(str(timenow.split()[0]).split("-")[1])
  day = int(str(timenow.split()[0]).split("-")[2])
  hours = int(str(timenow.split()[1]).split(":")[0])
  minutes = str(timenow.split()[1]).split(":")[1]
  indice=0
  print((hours > 12) and (hours < 18))
  if ((hours > 6) and (hours < 12)):
   indice=1
  if ((hours >= 12) and (hours < 18)):
   indice=2
  if ((hours >= 18)):
   indice=3
  return indice
  

def FA_shift():
 Shift=Calendar()
 indice=On_duty()
 week_number=datetime.utcnow().isocalendar()[1]
 return Shift[week_number,indice]



from audioop import avg
from unicodedata import unidata_version
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

# A loto basic simulation
# 5 numeros + 2 numéros mystères
import math

def parmi(k,n):
    return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))

# https://code-examples.net/fr/q/1ffdbd
def stirling(n):
    # http://en.wikipedia.org/wiki/Stirling%27s_approximation
    return math.sqrt(2*math.pi*n)*(n/math.e)**n

def npr(n,r):
    return (stirling(n)/stirling(n-r) if n>20 else
            math.factorial(n)/math.factorial(n-r))

def ncr(n,r):    
    return (stirling(n)/stirling(r)/stirling(n-r) if n>20 else
            math.factorial(n)/math.factorial(r)/math.factorial(n-r))

def low_ncr(n,r):
    R = 1.0
    D = 1.0
    if(r <100):
        for k in range(r):
            R = R * (n - k)
            if(k>0):
                D = D * (k+1)
    else :
        return False
    return R/D
etoile_plus = 0

def calculate_probs(etoileplus) : 
    prix_ticket = 2.5 + etoileplus
    Nboules = 50
    Netoiles = 12

    boules_tirage = 5
    etoile_tirage = int(2 - etoileplus)

    total_combinaisons = parmi(boules_tirage,Nboules)*parmi(etoile_tirage,Netoiles)

    


    tirages_semaine = 2

    # source  :https://www.euro-millions.com/fr/statistiques (consulte 29/04/22)
    ticket_sales_6_months= [19970544,28104215,22676021,30799931,25130636,34405908,27664296,37225002,18241443,41239142,29155502,24483234,18722486,25645674,
                        20501800,30481083,21576011,31948367,17289164,24425493,19572232,26552525,21162108,47560681,18989292,26823632,21024077,47409273,
                        18004271,25323777,17317447,24338721,19226396,25728250,19593681,27489351,17045518,23827756,18046252,24764524,16150195,24742337,
                        17641052,25504163,19295594,26326841,20765788,27222309,22601255,34019865,29338626]
    jckpots_6_months = 1e6*np.array([67,80,90.5,105,116.5,132,145,162,17,130,143,17,27,41,52.5,69.5,80,17.5,17,30,41,56,67,26,36.5,51,63,130,17,31,17,31,41,55,66,79,17,30,40,53,17,30.5,40,54,65,77,87,100,110,125,139])
    if (etoileplus>0):
        jckpots_6_months = 415000
    def prob_k_winner(k,prob_win,Nplayers):
        """Only works for k relatively small"""
        # If there are exactly k winners, it means that k player won and Nplayers-k players lost :
        #proba(1 seul gagnant) = proba(X gagne)**1 x proba(Xbar ne gagne pas) 
        r = 1.0
        r = low_ncr(Nplayers,k) * np.power(prob_win,k) * np.power(1-prob_win,Nplayers-k)
        return r

    N = int(np.mean(ticket_sales_6_months))
    J = int(np.mean(jckpots_6_months))



    pwin = 1/total_combinaisons
    sum = 0
    L = []
    I = 15
    for i in range(I):
        P = (prob_k_winner(i,pwin,N)) # Marginal probability
        sum  += P
        L.append(P)
    L = np.array(L)

    Psup2 = 1 - L[0] - L[1] # Probability of having two or more winners
    Psup1 = 1 - L[0]         # Probability of having one or more winners

    Pshare = Psup2/Psup1  # Probability of having another winner IF YOU HAVE WON
    print(J)

    mean_winner_gains = 0
    mean_winner_eff = 0
    weighted_gain = np.zeros(L.shape)
    for i in range(1,I):
        weighted_gain[i] = L[i]*J/i
        # If you ARE a winner, your gains will be shared with other winners 
        # MEAN INDIVIDUAL GAIN IF YOU HAVE WON : 
        mean_winner_gains += L[i]*J/i
        mean_winner_eff += L[i]*i
    mean_winner_gains = mean_winner_gains/Psup1
    #mean_winner_eff = mean_winner_eff/Psup1
    

    jackpot_ratio = 0.5  #https://www.euro-millions.com/fr/lots
    unitary_jackpot = 17*1e6
    
    
    income = prix_ticket*N
    total_jackpot = unitary_jackpot/jackpot_ratio
    
    other_costs_part_in_weekly_budget = 0.4
    lots_parts_in_weekly_budget = 1 - other_costs_part_in_weekly_budget
    other_costs = (total_jackpot/lots_parts_in_weekly_budget)*other_costs_part_in_weekly_budget
    print(other_costs)
    
    total_costs = total_jackpot +other_costs
    estimate_benefice_bank = income - total_costs
    

    
    
    if (etoileplus>0):
        print("ETOILE PLUS")
    else : 
        print("CLASSIC")
    print('---------------------------------------------------------')
    print(str(int(total_combinaisons)) + " possible combinations.")
    print()
    print("Mean total gain of all winners for a specific lottery : " + str(np.round(np.sum(weighted_gain)/1e6,1)) + " Million €")
    print()
    print("Individual expectancy : the average participant will gain " + str(np.round(pwin*J - (1-pwin)*prix_ticket,2)) + " € out of an invested "+ str(np.round(prix_ticket,2)) + " € .")
    print()
    print("If you are a super winner, your mean gains should be " + str(np.round(mean_winner_gains/1e6,2)) + " Million €, out of a " + str(np.round(J/1e6,2)) + " Million € pool.")
    print()
    print("On average, " + str(np.round(mean_winner_eff,2)) + " people will win the highest prize in this lottery.")
    print()
    print("On average, the lottery will make around " + str(np.round(estimate_benefice_bank/1e6,2)) + " Million € in raw benefits (income " + str(np.round(income/1e6,2)) + " m€ - estimated costs "+ str(np.round(total_costs/1e6,2)) + " m€ ).")

calculate_probs(1)
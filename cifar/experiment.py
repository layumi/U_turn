import os

test_rate = (2,4,8,12,16)
for i in range(5):
    rate = test_rate[i]
    for j in range(5):
        method_id = j+1
        print('------Rate %d Method:%d------'%(rate,method_id) )
        with open("Output.txt", "a") as text_file:
           text_file.write("|%d | %d | " % (rate, method_id))
        os.system('python attack.py --method_id %d --rate %d >> Output.txt'% (method_id, rate))

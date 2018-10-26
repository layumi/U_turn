import os
import evaluate_gpu

test_rate = (2,4,8,12,16)
for i in range(5):
    rate = test_rate[i]
    for j in range(5):
        method_id = j+1
        if method_id ==4:
            continue
        print('------Rate %d Method:%d------'%(rate,method_id) )
        os.system('python3 generate_attack_query.py --method_id %d --rate %d'% (method_id, rate))
        os.system('python3 prepare_attack.py --method_id %d --rate %d'% (method_id, rate))
        #if (i==0) and (j==0):
        #    os.system('python3 test_only_query.py --name PCB-8 --test_all --test_dir ./attack_query/pytorch/')
        #else:
        os.system('python3 test_only_query.py --name PCB-8 --test_dir ./attack_query/pytorch/')
        result = evaluate_gpu.main()
        with open("Output.txt", "a") as text_file:
            text_file.write("|%d | %d | %f | %f | %f | %f |\n" % (rate, method_id, 
                            result[0],result[1], result[2], result[3]))

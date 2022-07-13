import os
import evaluate_gpu
import time

os.environ['MKL_THREADING_LAYER'] = 'GNU'

#test_rate = (2,4,8,12,16)
test_rate = (16,12,8,4,2)
for i in range(1):
    rate = test_rate[i]
    #for j in range(8):
    for j in range(7):
        method_id = j+1
        if method_id ==4:
            continue
        print('------Rate %d Method:%d------'%(rate,method_id) )
        time_start=time.time()
        os.system('python3 generate_attack_query.py --randomstart --method_id %d --rate %d'% (method_id, rate))
        time_end=time.time()
        print('time cost',time_end-time_start,'s')
        os.system('python3 prepare_attack.py --method_id %d --rate %d'% (method_id, rate))
        if (i==0) and (j==0):
            os.system('python3 test_only_query.py --name baseline --test_all --test_dir ./attack_query/pytorch/')
        else:
            output_path = './attack_query/baseline-' + str(method_id) + '/' + str(rate) + '/'
            os.system('python3 test_only_query.py --name baseline --test_dir ./attack_query/pytorch/ --output_path %s'%output_path)
        result = evaluate_gpu.main()
        with open("Output_RS.txt", "a") as text_file:
            text_file.write("|%d | %d | %f | %f | %f | %f |\n" % (rate, method_id, 
                            result[0],result[1], result[2], result[3]))

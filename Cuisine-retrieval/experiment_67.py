import os
import evaluate_gpu

os.environ['MKL_THREADING_LAYER'] = 'GNU'
#test_rate = (16, 12, 8, 4, 2)
test_rate = (2,4,8,12,16)

query_save_path = 'attack_query'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)

for i in [-1]:
    rate = test_rate[i]
    for j in [5,6]:
        method_id = 1+j #6+j #j+1
        if method_id == 4:
            continue
        print('------Rate %d Method:%d------'%(rate,method_id) )
        os.system('python generate_attack_query.py --lr 1e-4 --iter 100 --method_id %d --rate %d --gpu_ids 0'% (method_id, rate))
        os.system('python prepare_attack.py --method_id %d --rate %d'% (method_id, rate))
        if (i==0) and (j==0):
         #   we need gallery feature
            os.system('python test_query.py --name ft_ResNet50_all --test_all --test_dir ./attack_query/pytorch/')
        else:
            output_path = './attack_query/ft_ResNet50_all-' + str(method_id) + '/' + str(rate) + '/'
            os.system('python test_query.py --name ft_ResNet50_all --test_dir %s --output_path %s --gpu_ids 0'%(output_path,output_path))
        result = evaluate_gpu.main(output_path)
        with open("Output_67.txt", "a") as text_file:
            text_file.write("|%d | %d | %.2f | %.2f | %.2f | %.2f |\n" % (rate, method_id, 
                            result[0]*100,result[1]*100, result[2]*100, result[3]*100))

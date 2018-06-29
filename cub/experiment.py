import os
import evaluate_gpu

test_rate = (2,4,8,12,16)
for i in range(5):
    rate = test_rate[i]
    for j in range(5):
        method_id = j+1
        if method_id == 4:
            continue
        print('------Rate %d Method:%d------'%(rate,method_id) )
        os.system('python generate_attack_query.py --method_id %d --rate %d --gpu_ids 1'% (method_id, rate))
        #os.system('python prepare_attack.py --method_id %d --rate %d'% (method_id, rate))
        #if (i==0) and (j==0):
         #   we need gallery feature
         #   os.system('python test_only_query.py --name ft_ResNet50 --test_all --test_dir ./attack_query/pytorch/')
        #else:
        output_path = './attack_query/ft_ResNet50_all-' + str(method_id) + '/' + str(rate) + '/'
        os.system('python test_query.py --name ft_ResNet50_all --test_dir %s --output_path %s --gpu_ids 1'%(output_path,output_path))
        result = evaluate_gpu.main(output_path)
        with open("Output.txt", "a") as text_file:
            text_file.write("|%d | %d | %.2f | %.2f | %.2f | %.2f |\n" % (rate, method_id, 
                            result[0]*100,result[1]*100, result[2]*100, result[3]*100))

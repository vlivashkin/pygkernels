from _generated_kkmeans import generated_kkmeans
from _generated_kward import generated_kward
from _datasets_kkmeans import datasets_kkmeans
from _datasets_kward import datasets_kward
from part2 import calc_part2
from part3 import calc_part3
from part4 import calc_part4
from part5 import calc_part5
from part6 import calc_part6

if __name__ == '__main__':
    N_JOBS = 16
    
    print('### CALC DATASETS KWARD ###')
    datasets_kward(n_jobs=N_JOBS)
    
#     print('### CALC CLASSIC PLOTS KKMEANS ###')
#     generated_kkmeans(n_jobs=N_JOBS)

    print('### DONE ###')
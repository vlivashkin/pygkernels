from _datasets_kkmeans import datasets_kkmeans_any
from _datasets_kward import datasets_kward
from _generated_kkmeans import generated_kkmeans_any
from _generated_kward import generated_kward
from part2 import calc_part2
from part3_KKMeans import calc_part3
from part4 import calc_part4
from part5 import calc_part5
from part6 import calc_part6

if __name__ == '__main__':
    N_JOBS = 6

    print('### CALC CLASSIC PLOTS KKMEANS ANY ###')
    generated_kkmeans_any(n_jobs=N_JOBS)
    print('### CALC CLASSIC PLOTS KWARD ###')
    generated_kward(n_jobs=N_JOBS)
    print('### CALC DATASETS KKMEANS ANY ###')
    datasets_kkmeans_any(n_jobs=N_JOBS)
    print('### CALC DATASETS KWARD ###')
    datasets_kward(n_jobs=N_JOBS)

    print('### CALC PART2')
    calc_part2(n_jobs=N_JOBS)
    print('### CALC PART3')
    calc_part3(n_jobs=N_JOBS)
    print('### CALC PART4')
    calc_part4()
    print('### CALC PART5')
    calc_part5(n_jobs=N_JOBS)
    print('### CALC PART6')
    calc_part6(n_jobs=N_JOBS)

    print('### DONE ###')

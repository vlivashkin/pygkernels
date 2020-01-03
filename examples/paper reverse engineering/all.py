from _classic_plot_kkmeans import classic_plots_kkmeans
from _classic_plot_kward import classic_plots_kward
from part2 import calc_part2
from part3 import calc_part3
from part4 import calc_part4
from part5 import calc_part5
from part6 import calc_part6

if __name__ == '__main__':
    N_JOBS = 6

    print('### CALC_CLASSIC_GRAPHS KKMEANS ###')
    classic_plots_kkmeans(n_jobs=N_JOBS)

    print('### CALC_CLASSIC_GRAPHS KWARD ###')
    classic_plots_kward(n_jobs=N_JOBS)

    print('### PART2 ###')
    calc_part2(n_jobs=N_JOBS)
    print('### PART3 ###')
    calc_part3(n_jobs=N_JOBS)
    print('### PART4 ###')
    calc_part4()
    print('### PART5 ###')
    calc_part5(n_jobs=N_JOBS)
    print('### PART6 ###')
    calc_part6(n_jobs=N_JOBS)

import pandas as pd
import numpy as np


for i in range(201):
    file_path = 'train_data/js_new/traj_{}_js_new.csv'.format(i)
    col_names = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6']
    # train_data = read_csv("../train_data/from_ge/Curve_js2.csv", names=col_names)
    data = pd.read_csv(file_path, names=col_names)
    # train_data = read_csv("../train_data/from_Jon/qbestcurve_new.csv", names=col_names)
    # train_data = read_csv("../constraint_solver/single_arm/trajectory/curve_pose_opt/curve_pose_opt_js.csv", names=col_names)
    # train_data = read_csv("../constraint_solver/single_arm/trajectory/all_theta_opt_blended/all_theta_opt_js.csv", names=col_names)
    # train_data = read_csv("../constraint_solver/single_arm/trajectory/init_opt/init_opt_js.csv", names=col_names)
    curve_q1 = data['q1'].tolist()
    curve_q2 = data['q2'].tolist()
    curve_q3 = data['q3'].tolist()
    curve_q4 = data['q4'].tolist()
    curve_q5 = data['q5'].tolist()
    curve_q6 = data['q6'].tolist()
    curve_js = np.vstack((curve_q1, curve_q2, curve_q3, curve_q4, curve_q5, curve_q6)).T

    new_curve_js = curve_js[::100]
    new_curve_js = np.vstack([new_curve_js, curve_js[-1]])
    new_file_path = 'train_data/js_new_500/traj_{}_js_new.csv'.format(i)
    # np.savetxt(new_file_path, new_curve_js, delimiter=',')
    df = pd.DataFrame(new_curve_js)
    df.to_csv(new_file_path, index=False, header=None)
    print("{} / {}".format(i, 201))

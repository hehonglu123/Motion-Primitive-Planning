########
# This module utilized https://github.com/johnwason/abb_motion_program_exec
# and send whatever the motion primitives that algorithms generate
# to RobotStudio
########

import time
from robotstudio_send import MotionSend
import psutil
import numpy as np
import csv

def main():

    # parameters
    max_iter = 24000
    # max_iter = 20
    save_every = 500
    show_status_every = 50

    ms = MotionSend()
    start_stamp = time.time()

    all_cpu = np.array([])
    all_ram = np.array([])
    errors_ave = np.array([])

    for i in range(max_iter):
        
        # if i % save_every == 0 or i == max_iter-1:
        #     ms.exec_motions_from_file(data_file_name='command_backproj.csv',\
        #         save_file=True,save_file_name='logged_joints/repeat_test/log_'+str(i)+'.csv')
        # else:
        #     ms.exec_motions_from_file(data_file_name='command_backproj.csv')

        if i==0:
            ms.exec_motions_from_file(data_file_name='command_backproj.csv',\
                    save_file=True,save_file_name='logged_joints/repeat_test/log_first.csv')
        else:
            ms.exec_motions_from_file(data_file_name='command_backproj.csv',\
                    save_file=True,save_file_name='logged_joints/repeat_test/log_temp.csv')
            
            with open("logged_joints/repeat_test/log_first.csv","r") as f:
                rows = csv.reader(f, delimiter=',')

                log_results_dict = {}
                for col in rows:
                    if len(log_results_dict) == 0:
                        log_results_dict['timestamp']=[]
                        log_results_dict['cmd_num']=[]
                        log_results_dict['joint_angle']=[]
                        continue
                    log_results_dict['timestamp'].append(float(col[0]))
                    log_results_dict['cmd_num'].append(int(col[1]))
                    log_results_dict['joint_angle'].append(np.deg2rad(np.array([float(col[2]),float(col[3]),float(col[4]),float(col[5]),float(col[6]),float(col[7])])))
                joint_angles1 = np.array(log_results_dict['joint_angle'])

            with open("logged_joints/repeat_test/log_temp.csv","r") as f:
                rows = csv.reader(f, delimiter=',')

                log_results_dict = {}
                for col in rows:
                    if len(log_results_dict) == 0:
                        log_results_dict['timestamp']=[]
                        log_results_dict['cmd_num']=[]
                        log_results_dict['joint_angle']=[]
                        continue
                    log_results_dict['timestamp'].append(float(col[0]))
                    log_results_dict['cmd_num'].append(int(col[1]))
                    log_results_dict['joint_angle'].append(np.deg2rad(np.array([float(col[2]),float(col[3]),float(col[4]),float(col[5]),float(col[6]),float(col[7])])))
                joint_angles2 = np.array(log_results_dict['joint_angle'])

            errors = np.array([])
            for j in range(min(len(joint_angles1),len(joint_angles2))):
                err = np.linalg.norm(joint_angles1[j]-joint_angles2[j])
                errors = np.append(errors,err)
            
            errors_ave = np.append(errors_ave,np.average(errors))
            with open('logged_joints/repeat_test/error_ave.npy','wb') as f:
                np.save(f, errors_ave)
        
        cpu_per = psutil.cpu_percent()
        ram_per = psutil.virtual_memory().percent

        if i % show_status_every == 0 or i == max_iter-1:
            now_stamp = time.time()
            dura = now_stamp-start_stamp
            m, s = divmod(dura, 60)
            h, m = divmod(m, 60)
            print("Iteration:",i+1)
            print("Estimated time:",h,"hours",m,"minutes",s,"seconds")
            print("CPU:",str(cpu_per),"%, RAM:",str(ram_per),'%')
            print("=========================================")
        
        all_cpu = np.append(all_cpu,cpu_per)
        all_ram = np.append(all_ram,ram_per)
        with open('logged_joints/repeat_test/cpu.npy','wb') as f:
            np.save(f, all_cpu)
        with open('logged_joints/repeat_test/ram.npy','wb') as f:
            np.save(f, all_ram)

if __name__ == "__main__":
    main()

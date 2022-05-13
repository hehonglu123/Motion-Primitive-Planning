from greedy import *



def main():
	###list of threshold metric
	thresholds=[0.1,0.2,0.5,0.9]
	data_dir='../data/wood/'
	###read in points
	curve_js = read_csv(data_dir+"Curve_js.csv",header=None).values

	robot=abb6640(d=50)

	for threshold in thresholds:
		greedy_fit_obj=greedy_fit(robot,curve_js,0.2)


		###set primitive choices to moveL only
		greedy_fit_obj.primitives={'movel_fit':greedy_fit_obj.movel_fit_greedy}
		breakpoints,primitives_choices,points=greedy_fit_obj.fit_under_error()

	

		############insert initial configuration#################
		primitives_choices.insert(0,'movej_fit')
		points.insert(0,[greedy_fit_obj.curve_fit_js[0]])

		###save commands
		df=DataFrame({'breakpoints':breakpoints,'primitives':primitives_choices,'points':points})
		df.to_csv(data_dir+str(threshold)+'/command.csv',header=True,index=False)
		df=DataFrame({'x':greedy_fit_obj.curve_fit[:,0],'y':greedy_fit_obj.curve_fit[:,1],'z':greedy_fit_obj.curve_fit[:,2],\
			'R1':greedy_fit_obj.curve_fit_R[:,0,0],'R2':greedy_fit_obj.curve_fit_R[:,0,1],'R3':greedy_fit_obj.curve_fit_R[:,0,2],\
			'R4':greedy_fit_obj.curve_fit_R[:,1,0],'R5':greedy_fit_obj.curve_fit_R[:,1,1],'R6':greedy_fit_obj.curve_fit_R[:,1,2],\
			'R7':greedy_fit_obj.curve_fit_R[:,2,0],'R8':greedy_fit_obj.curve_fit_R[:,2,1],'R9':greedy_fit_obj.curve_fit_R[:,2,2]})
		df.to_csv(data_dir+str(threshold)+'/curve_fit.csv',header=True,index=False)
		DataFrame(greedy_fit_obj.curve_fit_js).to_csv(data_dir+str(threshold)+'/curve_fit_js.csv',header=False,index=False)

if __name__ == "__main__":
	main()
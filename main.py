import csv
import sys
import time
import pandas as pd

def main_loop():
	while True:
		filename = "raw_numbers.csv"
		# opening the file with w+ mode truncate
		f = open(filename, "w+")
		f.close()

		filename1 = "radar_file.csv"
		# opening the file with w+ mode truncates the file
		f1 = open(filename1, "w+")
		f1.close()
		#time to flush files and restart
		time.sleep(10)

		df_lp = pd.read_csv('raw_numbers.csv', sep=',', header=None)
		lp_time, lp_number = df_lp[0], df_lp[1]

		df_mph = pd.read_csv('radar_file.csv', sep=',', header=None)
		mph_time, mph_number = df_mph[0], df_mph[1]
		for i, lp in df_lp.iterrows():
			time1 = lp[0]
			value1 = lp[1]
			for j, mph in df_mph.iterrows():
				#print("HERE")
				time2 = mph[0]
				value2 = mph[1]
				#print(time1,time2)
				if time1 == time2:
					eq_time1 = time1
					eq_time2 = time2
					print(f'Match at {time1} , {time2} with velocity of {value2} and license plate {value1}!')
					with open('log_file.csv', mode='a') as log_file:
						log_writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
						log_writer.writerow([f'{eq_time1}',f'{value1}',f'{value2}'])

if __name__ == '__main__':
	while(1):	
		try:
			main_loop()
		except KeyboardInterrupt:
			break 
		except:
			print("Empty Detect")

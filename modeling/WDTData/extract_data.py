import os
import pandas as pd
import sys
from sklearn.model_selection import train_test_split


input_dir = "Input/"

output_dir = "Output/"
output_training_dir = output_dir + "Training/"
output_training_data_dir = output_training_dir + "Data/"
output_test_dir = output_dir + "Test/"
output_test_data_dir = output_test_dir + "Data/"


def load_data():

	data = {}
	
	for filename in os.listdir(input_dir):
		if filename.endswith(".csv"):
			if filename.split(".")[0] != "att_2":
				data[filename.split(".")[0]] = pd.read_csv(input_dir + filename, encoding='UTF-16')
			elif filename.split(".")[0] == "att_2":
				data[filename.split(".")[0]] = pd.read_csv(input_dir + filename, encoding='UTF-16', sep="\t")
				
			if filename.split(".")[0] != "norm":
				data[filename.split(".")[0]] = data[filename.split(".")[0]].rename(columns={'TimeCol': 'Time'})
				temp = list(data[filename.split(".")[0]]["Time"])
				for idx,elem in enumerate(temp):
					temp[idx] = elem.split(" ")[1]
					temp[idx] = temp[idx].split(".")[0]
				data[filename.split(".")[0]]["Time"] = temp
				
		'''
			if filename.split(".")[0] != "att_4":
				data[filename.split(".")[0]] = pd.read_csv(input_dir + filename, encoding='UTF-16')
			elif filename.split(".")[0] == "att_4":
				data[filename.split(".")[0]] = pd.read_csv(input_dir + filename)
		'''
	return data
	
def extract_cycles(data):

	process_sensors = ["Time", "Tank_1", "Tank_4", "Tank_5", "Tank_6", "Tank_7", "Tank_8", "Pump_2", "Pump_4", "Pump_5", "Pump_6", "Valv_20", "Valv_22", "Flow_sensor_1", "Flow_sensor_2", "Flow_sensor_3"]
	all_processes_sensors = []

	temp = {}
	temp["N"] = []
	temp["A"] = []

	n_idx = 0
	a_idx = 0

	for sample in data:
		cycles = []
		index_interval = []
			
		tank_1_observations = list(data[sample]["Tank_1"].values)
		more_cycles = True
		while more_cycles == True:
			temp = pd.DataFrame(columns=list(data[sample].columns))
			end_cycle = False
			start_cycle_idx = 0
			while end_cycle == False:
				if len(tank_1_observations) > 1 and start_cycle_idx < 35:
					temp = pd.concat([temp, data[sample].head(1)], axis=0, ignore_index = True)
					data[sample].drop(index=data[sample].index[0], axis=0, inplace=True)
					del tank_1_observations[0]
					start_cycle_idx = start_cycle_idx + 1
				elif len(tank_1_observations) > 1 and tank_1_observations[0] > 5 and start_cycle_idx == 35:
					temp = pd.concat([temp, data[sample].head(1)], axis=0, ignore_index = True)
					data[sample].drop(index=data[sample].index[0], axis=0, inplace=True)
					del tank_1_observations[0]
				elif len(tank_1_observations) > 1 and tank_1_observations[0] < 5 and start_cycle_idx >= 35:
					temp = pd.concat([temp, data[sample].head(1)], axis=0, ignore_index = True)
					data[sample].drop(index=data[sample].index[0], axis=0, inplace=True)
					del tank_1_observations[0]
					start_cycle_idx = start_cycle_idx + 1
				else:
					end_cycle = True
			if len(temp)>50:
				cycles.append(temp)
			if len(tank_1_observations) > 1:
				more_cycles = True
			else:
				more_cycles = False	

		for idx,cycle in enumerate(cycles):
			cycles[idx] = cycles[idx][cycle.columns.intersection(process_sensors)]
			
		cycles.pop(0)
		data[sample] = cycles


	return data

def save_cycles(cycles, observation):

	if observation == "norm_training":
		isExist = os.path.exists(output_training_data_dir + "N")
		if not isExist:
			os.makedirs(output_training_data_dir + "N")
		for idx,cycle in enumerate(cycles):
			cycle.to_csv(output_training_data_dir + "N" + "/" + "N" + "_" + str(idx) + ".csv", index = False)

	elif observation == "norm_test_modeling":
		isExist = os.path.exists(output_test_data_dir + "N_modeling")
		if not isExist:
			os.makedirs(output_test_data_dir + "N_modeling")
		for idx,cycle in enumerate(cycles):
			cycle.to_csv(output_test_data_dir + "N_modeling" + "/" + "N" + "_" + str(idx) + ".csv", index = False)

	elif observation == "norm_test":
		isExist = os.path.exists(output_test_data_dir + "N")
		if not isExist:
			os.makedirs(output_test_data_dir + "N")
		for idx,cycle in enumerate(cycles):
			cycle.to_csv(output_test_data_dir + "N" + "/" + "N" + "_" + str(idx) + ".csv", index = False)

	elif observation.find("att") != -1:
		isExist = os.path.exists(output_test_data_dir + "A_" + observation.split("_")[-1])
		if not isExist:
			os.makedirs(output_test_data_dir + "A_" + observation.split("_")[-1])

		for idx,cycle in enumerate(cycles):
			cycle.to_csv(output_test_data_dir + "A_" + observation.split("_")[-1] + "/" + "A_" + observation.split("_")[-1] + "_" + str(idx) + ".csv", index = False)
	

	return None

data = load_data()
data = extract_cycles(data)

norm_train, norm_test_modeling = train_test_split(data["norm"],test_size=0.5)
save_cycles(norm_train, "norm_training")
save_cycles(norm_test_modeling, "norm_test_modeling")

norm_test = []
norm_test.append(data["att_1"][0]) # ciclo 1
norm_test.append(data["att_1"][2]) # ciclo 3
norm_test.append(data["att_1"][3]) # ciclo 4
norm_test.append(data["att_1"][4]) # ciclo 5
norm_test.append(data["att_1"][5]) # ciclo 6
norm_test.append(data["att_1"][6]) # ciclo 7
data["att_1"] = [data["att_1"][1]] # ciclo 2

norm_test.append(data["att_2"][0]) # ciclo 1
norm_test.append(data["att_2"][5]) # ciclo 6
data["att_2"] = [data["att_2"][1], data["att_2"][2], data["att_2"][3], data["att_2"][4]] # cicli 2, 3, 4 e 5

norm_test.append(data["att_3"][0]) # ciclo 1
norm_test.append(data["att_3"][3]) # ciclo 4
norm_test.append(data["att_3"][4]) # ciclo 5
norm_test.append(data["att_3"][5]) # ciclo 6
norm_test.append(data["att_3"][6]) # ciclo 7
norm_test.append(data["att_3"][7]) # ciclo 8
data["att_3"] = [data["att_3"][1], data["att_3"][2]] # cicli 2 e 3

norm_test.append(data["att_4"][0]) # ciclo 1
norm_test.append(data["att_4"][3]) # ciclo 4
norm_test.append(data["att_4"][4]) # ciclo 5
norm_test.append(data["att_4"][5]) # ciclo 6
norm_test.append(data["att_4"][6]) # ciclo 7
data["att_4"] = [data["att_4"][1], data["att_4"][2]] # cicli 2 e 3

save_cycles(norm_test, "norm_test")
save_cycles(data["att_1"], "att_1")
save_cycles(data["att_2"], "att_2")
save_cycles(data["att_3"], "att_3")
save_cycles(data["att_4"], "att_4")



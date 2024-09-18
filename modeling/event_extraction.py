import warnings
warnings.filterwarnings("ignore")

import os
import sys
import pandas as pd
from pandas.api.types import is_integer_dtype
from pandas.api.types import is_bool_dtype
import math
import random
from random import randint

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
import pm4py

from datetime import datetime


input_dir = "Input/EE/"
input_training_dir = input_dir + "Training/"
input_training_data_dir = input_training_dir + "Data/"

input_test_dir = input_dir + "Test/"
input_test_data_dir = input_test_dir + "Data/"
input_test_splitmetadata_dir = input_test_dir + "SplitMetaData/"

output_dir = "Output/EE/"
output_training_dir = output_dir + "Training/"
output_training_eventlogs_dir = output_training_dir + "EventLogs/"
output_training_splitmetadata_dir = output_training_dir + "SplitMetaData/"

output_test_dir = output_dir + "Test/"
output_test_eventlogs_dir = output_test_dir + "EventLogs/"

def load_data(phase):

	process_sensors = ["Time","Tank_1", "Tank_4", "Tank_5", "Tank_6", "Tank_7", "Tank_8"]

	if phase == "Training":
		input_data_dir = input_training_data_dir
	elif phase == "Test":
		input_data_dir = input_test_data_dir

	data = {}

	for observation in os.listdir(input_data_dir):
		for cycle in os.listdir(input_data_dir + observation):
			if cycle.endswith(".csv"):
				data[cycle.split(".")[0]] = pd.read_csv(input_data_dir + observation + "/" + cycle, usecols = process_sensors)
					
	'''
	for sample in data:
		for column in data[sample].columns:
			if column not in process_sensors:
				data[sample].drop(column, axis=1, inplace=True)
	'''
	return data
	
def search_state(value, quantization_intervals):
	
	state = None
	
	for label in quantization_intervals:
		if value >= quantization_intervals[label][0] and value <= quantization_intervals[label][1]:
			state = label
		
	if state == None:
		state = "OVERFLOW"

	return state
	
def quantize_data(timeseries, quantization_split_fraction, max_value, data_type, metadata, phase):

	if phase == "Training":

		if data_type == "numeric":
			quantized_timeseries = []
			quantization_intervals = {}
			quantization_step = math.floor(max_value/quantization_split_fraction)
			
			for i in range(0,quantization_split_fraction):
				if max_value > 0:
					if i < quantization_split_fraction-1:
						quantization_intervals[str(math.floor(1/quantization_split_fraction*(i)*100)) + "-" + str(math.floor(1/quantization_split_fraction*(i+1)*100))+"%"] = [i*quantization_step, (i+1)*quantization_step-1]
					else:
						quantization_intervals[str(math.floor(1/quantization_split_fraction*(i)*100)) + "-" + str(math.floor(1/quantization_split_fraction*(i+1)*100))+"%"] = [i*quantization_step, max_value]
				else:
					if i < quantization_split_fraction-1:
						quantization_intervals[str(math.floor(1/quantization_split_fraction*(i)*100)) + "-" + str(math.floor(1/quantization_split_fraction*(i+1)*100))+"%"] = [0, 0]
					else:
						quantization_intervals[str(math.floor(1/quantization_split_fraction*(i)*100)) + "-" + str(math.floor(1/quantization_split_fraction*(i+1)*100))+"%"] = [0, 0]
			
			for idx,value in enumerate(timeseries):
				quantized_timeseries.append(tuple([search_state(value, quantization_intervals),value]))
				
			return quantized_timeseries, quantization_intervals
			
		elif data_type == "boolean":
			quantized_timeseries = []
			for idx, value in enumerate(timeseries):
				if value == False:
					quantized_timeseries.append(tuple(["F",False]))
				elif value == True:
					quantized_timeseries.append(tuple(["T",True]))
			return quantized_timeseries

	elif phase == "Test":
		quantized_timeseries = []
		if data_type == "numeric":
			for idx,value in enumerate(timeseries):
				quantized_timeseries.append(tuple([search_state(value, metadata),value]))

		elif data_type == "boolean":
			quantized_timeseries = []
			for idx, value in enumerate(timeseries):
				if value == False:
					quantized_timeseries.append(tuple(["F",False]))
				elif value == True:
					quantized_timeseries.append(tuple(["T",True]))
		
		return quantized_timeseries
	
def per_sensor_split(data, metadata, quantization_split_fraction, phase):

	per_sensor_split_data = {}
	per_sensor_split_metadata = {}
	
	# Collect the per-sensor metadata for numerical sensors.
	max_values = {}
	for timeseries in data:
		for column in data[timeseries]:
			if column not in ["Time"]:
				if(is_integer_dtype(data[timeseries][column]) == True):
					try:
						max_values[column] = max(max(data[timeseries][column]),max_values[column])
					except:
						max_values[column] = max(data[timeseries][column])
		
	for timeseries in data:
		per_sensor_split_data[timeseries] = {}
		for column in data[timeseries]:
			if column not in ["Time"]:
				per_sensor_split_data[timeseries][column] = {}
				if(is_integer_dtype(data[timeseries][column]) == True):
					if per_sensor_split_metadata.get(column) == None:
						per_sensor_split_metadata[column] = []
					per_sensor_split_data[timeseries][column], per_sensor_split_metadata[column] = quantize_data(list(data[timeseries][column].values), quantization_split_fraction, max_values[column], "numeric")
				elif(is_bool_dtype(data[timeseries][column]) == True):
					per_sensor_split_data[timeseries][column] = quantize_data(list(data[timeseries][column].values),quantization_split_fraction, None, "boolean")
		
	return per_sensor_split_data, per_sensor_split_metadata
	
def build_dfs(split_data, data):

	
	dfs = {}
	
	for timeseries in data:
		df = pd.DataFrame(columns = list(data[timeseries].columns))
		for column in data[timeseries]:
				
			if column not in list(split_data[timeseries].keys()):
				df[column] = data[timeseries][column]
			else:
				df[column] = split_data[timeseries][column]
		dfs[timeseries] = df
	
	return dfs
	
def save_metadata(metadata, split_type):

	if split_type == "per_sensor":
		for metadata_set in metadata:
			file = open(output_training_splitmetadata_dir + metadata_set + ".txt", "w")
			for idx,level in enumerate(metadata[metadata_set]):
				if idx<len(metadata[metadata_set])-1:
					file.write(str(level) + ":" + str(metadata[metadata_set][level]) + "\n")
				else:
					file.write(str(level) + ":" + str(metadata[metadata_set][level]))
			file.close()
			
	elif split_type == "per_sensor_type":
		for sensor_type in metadata:
			file = open(output_training_splitmetadata_dir + sensor_type + ".txt", "w")
			for outer_idx,metadata_set in enumerate(metadata[sensor_type]):
				file.write(metadata_set + ":\n")
				for inner_idx,level in enumerate(metadata[sensor_type][metadata_set]):
					if inner_idx<len(metadata[sensor_type][metadata_set])-1:
						file.write(str(level) + ":" + str(metadata[sensor_type][metadata_set][level]) + "\n")
					else:
						file.write(str(level) + ":" + str(metadata[sensor_type][metadata_set][level]))
				if outer_idx<len(metadata[sensor_type])-1:		
					file.write("\n")
			file.close()
	
	return None
	
def convert_timestamp(timestamp):

	# old WDT data
	'''
	date = timestamp.split(" ")[0]
	time = timestamp.split(" ")[1]

	YYYY = date.split("/")[2]
	MM = date.split("/")[1]
	DD = date.split("/")[0]

	hh = time.split(":")[0]
	mm = time.split(":")[1]
	try:
		ss = time.split(":")[2]
	except:
		ss = "00"
	
	SSS = "000"
	'''

	# new WDT data
	time = timestamp

	YYYY = "1900"
	MM = "01"
	DD = "01"

	hh = time.split(":")[0]
	mm = time.split(":")[1]
	try:
		ss = time.split(":")[2]
	except:
		ss = "00"
	
	SSS = "000"

	converted_timestamp = YYYY + "-" + MM + "-" + DD + "T" + hh + ":" + mm + ":" + ss + "." + SSS

	return converted_timestamp

def build_event_log(dataset, timestamps, split_type, sensor_type, sensors_list, case_number):
	caseid = case_number
	event_log = []
	
	if split_type == "per_sensor":
		previous_state = dataset[0][0]
		for idx,instance in enumerate(dataset[1:]):
			current_state = instance[0]
			if current_state != previous_state:
				event_timestamp = convert_timestamp(timestamps[idx+1])
				event_value = instance[1]
				state_transition = str(previous_state)+"_to_"+str(current_state)
				event = [caseid, state_transition, event_value, event_timestamp]
				event_log.append(event)
				previous_state = current_state
			
		event_log = pd.DataFrame(event_log, columns=['CaseID', 'Event', 'Value', 'Timestamp'])
		event_log.rename(columns={'Event': 'concept:name'}, inplace=True)
		event_log.rename(columns={'Timestamp': 'time:timestamp'}, inplace=True)
		event_log = dataframe_utils.convert_timestamp_columns_in_df(event_log)
		parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'CaseID'}
		event_log = log_converter.apply(event_log, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
		
	elif split_type == "per_sensor_type":

		for sensor_idx,sensor in enumerate(dataset):
			previous_state = sensor[0][0]
			for idx,instance in enumerate(sensor[1:]):
				current_state = instance[0]
				if current_state == None:
					print(sensor[1:])
					sys.exit()
				if current_state != previous_state:
					event_timestamp = convert_timestamp(timestamps[idx+1])
					event_timestamp = datetime.strptime(event_timestamp, "%Y-%m-%dT%H:%M:%S.%f")
					event_value = instance[1]
					state_transition = str(sensors_list[sensor_idx]) + "-" + str(previous_state)+"_to_"+str(current_state)
					event = [caseid, state_transition, event_value, event_timestamp]
					event_log.append(event)
					previous_state = current_state
									
		event_log = pd.DataFrame(event_log, columns=['CaseID', 'Event', 'Value', 'Timestamp'])
		event_log.sort_values(by='Timestamp', inplace=True, ignore_index=True)
		timestamps = list(event_log["Timestamp"].values)
		for idx,timestamp in enumerate(timestamps):
			timestamps[idx] = str(timestamp)
		event_log["Timestamp"] = timestamps
		event_log.rename(columns={'Event': 'concept:name'}, inplace=True)
		event_log.rename(columns={'Timestamp': 'time:timestamp'}, inplace=True)
		event_log = dataframe_utils.convert_timestamp_columns_in_df(event_log)
		parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'CaseID'}
		event_log = log_converter.apply(event_log, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
	return event_log	
	
def extract_event_logs(quantized_data, split_type, per_sensor_type_split_data):

	event_data = {}

	unique_observations = {}
	for timeseries in quantized_data:
		observation = ""
		tokens = timeseries.split("_")
		for idx,token in enumerate(tokens[0:len(tokens)-1]):
			if idx < len(tokens[0:len(tokens)-1])-1:
				observation = observation + token + "_"
			else:
				observation = observation + token
		if observation not in list(unique_observations.keys()):
			unique_observations[observation] = []
			unique_observations[observation].append(quantized_data[timeseries])
		else:
			unique_observations[observation].append(quantized_data[timeseries])
	
	if split_type == "per_sensor":
		for timeseries in quantized_data:
			event_data[timeseries] = {}
			for column in quantized_data[timeseries]:
				if column != "Time":
					event_data[timeseries][column] = build_event_log(list(quantized_data[timeseries][column].values), list(quantized_data[timeseries]["Time"].values), split_type, None, int(timeseries.split("_")[-1]))
					
	elif split_type == "per_sensor_type":
		for timeseries in quantized_data:
			event_data[timeseries] = {}
			for sensor_type in list(per_sensor_type_split_data[timeseries].keys()):
				event_data[timeseries][sensor_type] = None
				sensor_type_quantized_data = []
				for sensor in per_sensor_type_split_data[timeseries][sensor_type]:
					sensor_type_quantized_data.append(list(quantized_data[timeseries][sensor].values))
					if timeseries == "A_0":
						list(quantized_data[timeseries][sensor].values)
				
				event_data[timeseries][sensor_type] = build_event_log(sensor_type_quantized_data, list(quantized_data[timeseries]["Time"].values), split_type, sensor_type, list(per_sensor_type_split_data[timeseries][list(per_sensor_type_split_data[timeseries].keys())[0]]), int(timeseries.split("_")[-1]))
				
		new_event_data = {}
		for unique_observation in unique_observations:
			
			temp = {}
			for timeseries in event_data:
				if timeseries.find(unique_observation) != -1:
					new_event_data[unique_observation] = {}
					for sensor_type in event_data[timeseries]:
						if sensor_type not in list(temp.keys()):
							temp[sensor_type] = []
							temp[sensor_type].append(event_data[timeseries][sensor_type])
						else:
							temp[sensor_type].append(event_data[timeseries][sensor_type])

			
			for sensor_type in temp:	
				new_event_data[unique_observation][sensor_type] = merge_event_logs(temp[sensor_type])
						
	return new_event_data
	
def merge_event_logs(event_logs):

	merged_event_log = []

	for event_log in event_logs:
		merged_event_log.append(event_log[0])
	
	merged_event_log = (pm4py.objects.log.obj.EventLog)(merged_event_log)

	return merged_event_log
	
def save_event_logs(event_data, phase):

	if phase == "Training":
		output_eventlogs_dir = output_training_eventlogs_dir
	elif phase == "Test":
		output_eventlogs_dir = output_test_eventlogs_dir

	for timeseries in event_data:
		if not os.path.exists(output_eventlogs_dir + timeseries):
			os.makedirs(output_eventlogs_dir + timeseries)
			
		for event_log in event_data[timeseries]:
			xes_exporter.apply(event_data[timeseries][event_log], output_eventlogs_dir + timeseries + "/" + event_log + '.xes')

	return None
	
def per_sensor_type_split(data, metadata, quantization_split_fraction, phase):

	if phase == "Training":
		per_sensor_type_split_data = {}
		per_sensor_type_split_metadata = {}
		sensor_types = {}
		for sensor_type in data[list(data.keys())[0]]:
			if sensor_type != "Time":
				try:
					sensor_types[sensor_type.split("_")[0]][sensor_type] = []
				except:
					sensor_types[sensor_type.split("_")[0]] = {}
					sensor_types[sensor_type.split("_")[0]][sensor_type] = []
		
		for timeseries in data:
			per_sensor_type_split_data[timeseries] = {}
			for sensor_type in sensor_types:
				per_sensor_type_split_data[timeseries][sensor_type] = sensor_types[sensor_type].copy()
				per_sensor_type_split_metadata[sensor_type] = sensor_types[sensor_type].copy()
				
		for sensor_type in sensor_types:		
			per_sensor_type_split_metadata[sensor_type] = sensor_types[sensor_type].copy()
			
			
		max_values = {}
		for timeseries in data:
			for column in data[timeseries]:
				if column not in ["Time"]:
					if(is_integer_dtype(data[timeseries][column]) == True):
						try:
							max_values[column]= max(max(data[timeseries][column]),max_values[column])
						except:
							max_values[column] = max(data[timeseries][column])
		
		per_sensor_split_data = {}
		per_sensor_split_metadata = {}
		for timeseries in data:
			
			per_sensor_split_data[timeseries] = {}
			for column in data[timeseries]:
				if column not in ["Time"]:
					per_sensor_split_data[timeseries][column] = {}
					if(is_integer_dtype(data[timeseries][column]) == True):
						per_sensor_split_data[timeseries][column], per_sensor_split_metadata[column] = quantize_data(list(data[timeseries][column].values), quantization_split_fraction, max_values[column], "numeric", None, phase)
					elif(is_bool_dtype(data[timeseries][column]) == True):
						per_sensor_split_data[timeseries][column] = quantize_data(list(data[timeseries][column].values),quantization_split_fraction, None, "boolean", None, phase)
			
			
			for column in per_sensor_split_data[timeseries]:
				per_sensor_type_split_data[timeseries][column.split("_")[0]][column] = per_sensor_split_data[timeseries][column]
				
		to_retain_sensor_type_metadata = []
		for column in per_sensor_split_metadata:
			if column.split("_")[0] not in to_retain_sensor_type_metadata:
				to_retain_sensor_type_metadata.append(column.split("_")[0])
		
		to_delete_sensor_type_metadata = []				
		for sensor_type in list(per_sensor_type_split_metadata.keys()):
			if sensor_type not in to_retain_sensor_type_metadata:
				to_delete_sensor_type_metadata.append(sensor_type)
				
		for sensor_type in to_delete_sensor_type_metadata:
			del per_sensor_type_split_metadata[sensor_type]
			
		for column in per_sensor_split_metadata:
			per_sensor_type_split_metadata[column.split("_")[0]][column] = per_sensor_split_metadata[column]

		return per_sensor_type_split_data, per_sensor_type_split_metadata, per_sensor_split_data

	elif phase == "Test":
		per_sensor_type_split_data = {}
		sensor_types = {}
		for sensor_type in data[list(data.keys())[0]]:
			if sensor_type != "Time":
				try:
					sensor_types[sensor_type.split("_")[0]][sensor_type] = []
				except:
					sensor_types[sensor_type.split("_")[0]] = {}
					sensor_types[sensor_type.split("_")[0]][sensor_type] = []
		
		for timeseries in data:
			per_sensor_type_split_data[timeseries] = {}
			for sensor_type in sensor_types:
				per_sensor_type_split_data[timeseries][sensor_type] = sensor_types[sensor_type].copy()

		per_sensor_split_data = {}
		for timeseries in data:
			per_sensor_split_data[timeseries] = {}
			for column in data[timeseries]:
				if column != "Time":
					per_sensor_split_data[timeseries][column] = {}
					if(is_integer_dtype(data[timeseries][column]) == True):
						per_sensor_split_data[timeseries][column] = quantize_data(list(data[timeseries][column].values), quantization_split_fraction, None, "numeric", metadata[column.split("_")[0]][column], phase)
					elif(is_bool_dtype(data[timeseries][column]) == True):
						per_sensor_split_data[timeseries][column] = quantize_data(list(data[timeseries][column].values),quantization_split_fraction, None, "boolean", None, phase)
		
			for column in per_sensor_split_data[timeseries]:
				per_sensor_type_split_data[timeseries][column.split("_")[0]][column] = per_sensor_split_data[timeseries][column]

		return per_sensor_type_split_data, per_sensor_split_data
	
def read_metadata(data, split_type):

	metadata = {}

	if split_type == "per_sensor":
		pass
	elif split_type == "per_sensor_type":
		dataset_features = list(data[list(data.keys())[0]].columns)
		dataset_features.remove("Time")
		sensor_types = []
		for sensor in dataset_features:
			sensor_type = sensor.split("_")[0]
			if sensor_type not in sensor_types:
				sensor_types.append(sensor_type)
		
		for sensor_type in sensor_types:
			metadata[sensor_type] = {}
			sensor_type_metadata_file = open(input_test_splitmetadata_dir + sensor_type + ".txt", "r")
			lines = sensor_type_metadata_file.readlines()
			current_sensor = ""
			for line in lines:
				tokens = line.split(":")
				if tokens[0].find(sensor_type) != -1:
					current_sensor = tokens[0]
					metadata[sensor_type][current_sensor] = {}
				else:
					metadata[sensor_type][current_sensor][tokens[0]] = []
					tokens[1] = tokens[1].replace("[","").replace("]","").strip("\n").strip()
					boundaries = tokens[1].split(",")
					metadata[sensor_type][current_sensor][tokens[0]].append(int(boundaries[0]))
					metadata[sensor_type][current_sensor][tokens[0]].append(int(boundaries[1]))
					
			

	return metadata

try:
	phase = sys.argv[1]
	split_type = sys.argv[2]
	quantization_split_fraction = int(sys.argv[3])
except IndexError:	
	print("Input the right number of arguments.")
	sys.exit()
	

data = load_data(phase)

if phase == "Training":
	if split_type == "per_sensor":
		per_sensor_split_data, per_sensor_split_metadata = per_sensor_split(data, quantization_split_fraction)
		save_metadata(per_sensor_split_metadata, split_type)
		quantized_data = build_dfs(per_sensor_split_data, data)
		event_data = extract_event_logs(quantized_data, split_type, None)
		
	elif split_type == "per_sensor_type":
		per_sensor_type_split_data, per_sensor_type_split_metadata, per_sensor_split_data = per_sensor_type_split(data, None, quantization_split_fraction, phase)
		save_metadata(per_sensor_type_split_metadata, split_type)
		quantized_data = build_dfs(per_sensor_split_data, data)
		event_data = extract_event_logs(quantized_data, split_type, per_sensor_type_split_data)
		
elif phase == "Test":
	if split_type == "per_sensor":
		pass
	elif split_type == "per_sensor_type":
		per_sensor_split_metadata = read_metadata(data, split_type)
		per_sensor_type_split_data, per_sensor_split_data = per_sensor_type_split(data, per_sensor_split_metadata, quantization_split_fraction, phase)
		quantized_data = build_dfs(per_sensor_split_data, data)
		event_data = extract_event_logs(quantized_data, split_type, per_sensor_type_split_data)

save_event_logs(event_data, phase)




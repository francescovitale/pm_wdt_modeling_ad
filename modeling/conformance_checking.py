from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
import pm4py.algo.conformance.alignments as alignments
import pm4py.algo.evaluation.replay_fitness as replay_fitness
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator

import pm4py
import sys
import os
import pandas as pd
import numpy as np
import time
import datetime

import func_timeout

input_dir = "Input/CC/"
input_test_dir = input_dir + "Test/"
input_test_eventlogs_dir = input_test_dir + "EventLogs/"
input_test_petrinets_dir = input_test_dir + "PetriNets/"

input_training_dir = input_dir + "Training/"
input_training_eventlogs_dir = input_training_dir + "EventLogs/"
input_training_petrinets_dir = input_training_dir + "PetriNets/"

output_dir = "Output/CC/"
output_test_dir = output_dir + "Test/"
output_test_metrics_dir = output_test_dir + "Metrics/"
output_test_diagnoses_dir = output_test_dir + "Diagnoses/"

output_training_dir = output_dir + "Training/"
output_training_diagnoses_dir = output_training_dir + "Diagnoses/"

fitness = {}
precision = {}
generalization = {}
simplicity = {}

def read_event_logs(mode, event_logs_to_check):

	event_logs = {}
	
	if mode == "Test":

		for type in os.listdir(input_test_eventlogs_dir):
			if event_logs_to_check == "N_only":
				if type == "N":
					event_logs[type] = {}
					for event_log_file in os.listdir(input_test_eventlogs_dir + type + "/"):
						event_logs[type][event_log_file.split(".xes")[0]] = xes_importer.apply(input_test_eventlogs_dir + type + "/" + event_log_file)
			elif event_logs_to_check == "A_only":
				if type != "N":
					event_logs[type] = {}
					for event_log_file in os.listdir(input_test_eventlogs_dir + type + "/"):
						event_logs[type][event_log_file.split(".xes")[0]] = xes_importer.apply(input_test_eventlogs_dir + type + "/" + event_log_file)
			elif event_logs_to_check == "All":
				event_logs[type] = {}
				for event_log_file in os.listdir(input_test_eventlogs_dir + type + "/"):
					event_logs[type][event_log_file.split(".xes")[0]] = xes_importer.apply(input_test_eventlogs_dir + type + "/" + event_log_file)

	elif mode == "Training":
		for type in os.listdir(input_training_eventlogs_dir):
			event_logs[type] = {}
			for event_log_file in os.listdir(input_training_eventlogs_dir + type + "/"):
				event_logs[type][event_log_file.split(".xes")[0]] = xes_importer.apply(input_training_eventlogs_dir + type + "/" + event_log_file)

	return event_logs

def read_petri_nets(mode):

	petri_nets = {}

	if mode == "Test":
		for type in os.listdir(input_test_petrinets_dir):
			petri_nets[type] = {}
			for petri_net in os.listdir(input_test_petrinets_dir + type + "/"):
				petri_nets[type][petri_net.split(".pnml")[0]] = {}
				petri_nets[type][petri_net.split(".pnml")[0]]["net"], petri_nets[type][petri_net.split(".pnml")[0]]["initial_marking"], petri_nets[type][petri_net.split(".pnml")[0]]["final_marking"] = pnml_importer.apply(input_test_petrinets_dir + type + "/" + petri_net)

	elif mode == "Training":
		for type in os.listdir(input_training_petrinets_dir):
			petri_nets[type] = {}
			for petri_net in os.listdir(input_training_petrinets_dir + type + "/"):
				petri_nets[type][petri_net.split(".pnml")[0]] = {}
				petri_nets[type][petri_net.split(".pnml")[0]]["net"], petri_nets[type][petri_net.split(".pnml")[0]]["initial_marking"], petri_nets[type][petri_net.split(".pnml")[0]]["final_marking"] = pnml_importer.apply(input_training_petrinets_dir + type + "/" + petri_net)

	return petri_nets

def compute_fitness(event_logs, petri_nets, cc_variant, cc_variant_type):

	aligned_traces = {}
	fitness = {}
	parameters ={}
	for type in event_logs:
		aligned_traces[type] = {}
		fitness[type] = {}
		for event_log in event_logs[type]:
			if cc_variant == "alignment_based":
				if cc_variant_type == "dijkstra_less_memory":
					aligned_traces[type][event_log] = alignments.petri_net.algorithm.apply_log(log = event_logs[type][event_log], petri_net = petri_nets[event_log]["net"], initial_marking = petri_nets[event_log]["initial_marking"], final_marking = petri_nets[event_log]["final_marking"], parameters = parameters, variant = alignments.petri_net.algorithm.Variants.VERSION_DIJKSTRA_LESS_MEMORY)
				elif cc_variant_type == "dijkstra_no_heuristics":
					aligned_traces[type][event_log] = alignments.petri_net.algorithm.apply_log(log = event_logs[type][event_log], petri_net = petri_nets[event_log]["net"], initial_marking = petri_nets[event_log]["initial_marking"], final_marking = petri_nets[event_log]["final_marking"], parameters = parameters, variant = alignments.petri_net.algorithm.Variants.VERSION_DIJKSTRA_NO_HEURISTICS)
				elif cc_variant_type == "discounted_a_star":
					aligned_traces[type][event_log] = alignments.petri_net.algorithm.apply_log(log = event_logs[type][event_log], petri_net = petri_nets[event_log]["net"], initial_marking = petri_nets[event_log]["initial_marking"], final_marking = petri_nets[event_log]["final_marking"], parameters = parameters, variant = alignments.petri_net.algorithm.Variants.VERSION_DISCOUNTED_A_STAR)
				elif cc_variant_type == "state_equation_a_star":
					aligned_traces[type][event_log] = alignments.petri_net.algorithm.apply_log(log = event_logs[type][event_log], petri_net = petri_nets[event_log]["net"], initial_marking = petri_nets[event_log]["initial_marking"], final_marking = petri_nets[event_log]["final_marking"], parameters = parameters, variant = alignments.petri_net.algorithm.Variants.VERSION_STATE_EQUATION_A_STAR)
				elif cc_variant_type == "tweaked_state_equation_a_star":
					aligned_traces[type][event_log] = alignments.petri_net.algorithm.apply_log(log = event_logs[type][event_log], petri_net = petri_nets[event_log]["net"], initial_marking = petri_nets[event_log]["initial_marking"], final_marking = petri_nets[event_log]["final_marking"], parameters = parameters, variant = alignments.petri_net.algorithm.Variants.VERSION_TWEAKED_STATE_EQUATION_A_STAR)
				fitness[type][event_log] = replay_fitness.algorithm.evaluate(results = aligned_traces[type][event_log], variant = replay_fitness.algorithm.Variants.ALIGNMENT_BASED)["log_fitness"]
			if cc_variant == "token_based":
				fitness[type][event_log] = 0
				trace_wise_results = pm4py.conformance_diagnostics_token_based_replay(event_logs[type][event_log], petri_nets[event_log]["net"], petri_nets[event_log]["initial_marking"], petri_nets[event_log]["final_marking"])
				for trace in trace_wise_results:
					fitness[type][event_log] = fitness[type][event_log] + trace["trace_fitness"]
				fitness[type][event_log] = fitness[type][event_log]/len(trace_wise_results)
			
	return fitness, aligned_traces

def compute_precision(event_logs, petri_nets):

	precision = {}

	for event_log in event_logs:
		precision[event_log] = pm4py.precision_alignments(event_logs[event_log], petri_nets[event_log]["net"], petri_nets[event_log]["initial_marking"], petri_nets[event_log]["final_marking"])

	return precision

def compute_generalization(event_logs, petri_nets):

	generalization = {}
	for event_log in event_logs:
		generalization[event_log] = generalization_evaluator.apply(event_logs[event_log], petri_nets[event_log]["net"], petri_nets[event_log]["initial_marking"], petri_nets[event_log]["final_marking"])

	return generalization

def compute_simplicity(petri_nets):

	simplicity = {}
	for petri_net in petri_nets:
		simplicity[petri_net] = simplicity_evaluator.apply(petri_nets[petri_net]["net"])

	return simplicity

def build_diagnoses(event_logs, petri_net, cc_variant, cc_variant_type):
	
	diagnoses_times = []
	average_cycle_time = 0
	n_traces = 0
	for type in event_logs:
		for event_log in event_logs[type]:
			for trace in event_logs[type][event_log]:
				n_traces = n_traces + 1
				average_cycle_time = average_cycle_time + (trace[-1]["time:timestamp"] - trace[0]["time:timestamp"]).total_seconds()
	average_cycle_time = average_cycle_time/n_traces

	if cc_variant == "alignment_based":
		diagnoses = None

		columns = []
		traces_activities = []
		petri_net_activities = []
		
		for type in event_logs:
			for event_log in event_logs[type]:
				traces_activities = traces_activities + get_event_log_activities(event_logs[type][event_log])
		
		for event_log in petri_net:
			petri_net_activities = petri_net_activities + get_petri_net_activities(petri_net[event_log])
		
		columns = traces_activities + petri_net_activities
		columns = list(set(columns))
		columns.sort()
		columns.append("Fitness")
		columns.append("Label")
		
		diagnoses = pd.DataFrame(columns = columns)	
		
		for type in event_logs:
			for event_log in event_logs[type]:
				aligned_traces = {}
				aligned_traces[type] = {}
				aligned_traces[type][event_log] = []
				
				for trace in event_logs[type][event_log]:
					temp = {}
					temp[type] = {}
					temp[type][event_log] = {}
					temp[type][event_log] = (pm4py.objects.log.obj.EventLog)([trace])
					initial_time = time.time()
					fitness, temp = compute_fitness(temp, petri_net, cc_variant, cc_variant_type)
					diagnoses_times.append(time.time() - initial_time)
					aligned_traces[type][event_log].append(temp[type][event_log][0])
				
				for trace in aligned_traces[type][event_log]:
					unaligned_activities = compute_unaligned_activities([trace])
					trace_fitness = trace["fitness"]
					new_row = {}
					for column in columns:
						if column not in list(unaligned_activities.keys()):
							new_row[column] = 0
						else:
							new_row[column] = unaligned_activities[column]
					new_row["Fitness"] = trace_fitness
					if type == "N":
						new_row["Label"] = "N"
					else:
						new_row["Label"] = "A"
					diagnoses = pd.concat([diagnoses, pd.DataFrame([new_row])], axis=0)
	elif cc_variant == "token_based":
		diagnoses = None

		columns = []
		petri_net_activities = []
		
		
		for event_log in petri_net:
			petri_net_activities = petri_net_activities + get_petri_net_activities(petri_net[event_log])
		
		columns = petri_net_activities
		columns = list(set(columns))
		columns.sort()
		columns.append("m")
		columns.append("r")
		columns.append("Fitness")
		columns.append("Label")
		
		diagnoses = pd.DataFrame(columns = columns)
		
		for type in event_logs:
			for event_log in event_logs[type]:
				for trace in event_logs[type][event_log]:
					row = {}
					for column in columns:
						row[column] = 0
					row["m"] = 0
					row["r"] = 0
					row["Fitness"] = 0
					initial_time = time.time()
					replayed_trace = pm4py.conformance_diagnostics_token_based_replay((pm4py.objects.log.obj.EventLog)([trace]), petri_net[event_log]["net"], petri_net[event_log]["initial_marking"], petri_net[event_log]["final_marking"])
					diagnoses_times.append(time.time() - initial_time)
					activated_transitions = replayed_trace[0]["activated_transitions"]
					for activated_transition in activated_transitions:
						if activated_transition.label is not None and activated_transition.label in columns:
							row[activated_transition.label] = row[activated_transition.label] + 1
					row["m"] = row["m"] + replayed_trace[0]["missing_tokens"]
					row["r"] = row["r"] + replayed_trace[0]["remaining_tokens"]		
					row["Fitness"] = replayed_trace[0]["trace_fitness"]
					if type == "N":
						row["Label"] = "N"
					else:
						row["Label"] = "A"
					
					
					diagnoses = pd.concat([diagnoses, pd.DataFrame(columns = columns, data = [list(row.values())])], ignore_index=True, axis=0)
	
	mean_diagnoses_time = sum(diagnoses_times)/len(diagnoses_times)
	std_diagnoses_time = np.std(diagnoses_times)

	diagnoses_time_percentage = mean_diagnoses_time/average_cycle_time
	diagnoses_std_time_percentage = std_diagnoses_time/average_cycle_time

	return diagnoses, mean_diagnoses_time, std_diagnoses_time, diagnoses_time_percentage, diagnoses_std_time_percentage
	
def get_event_log_activities(event_log):

	traces_activities = None
	
	traces_activities = list(pm4py.get_event_attribute_values(event_log, "concept:name").keys())

	return traces_activities
	
def get_petri_net_activities(petri_net):

	activities = []
	
	transitions = petri_net["net"].transitions
	for transition in transitions:
		if transition.label != None:
			activities.append(transition.label)

	return activities

def compute_unaligned_activities(aligned_traces):
	
	unaligned_activities = {}

	events = {}
	temp = []
	for aligned_trace in aligned_traces:
		temp.append(list(aligned_trace.values())[0])
	aligned_traces = temp
	for aligned_trace in aligned_traces:
		for move in aligned_trace:
			log_behavior = move[0]
			model_behavior = move[1]
			if log_behavior != model_behavior:
				if log_behavior != None and log_behavior != ">>":
					try:
						events[log_behavior] = events[log_behavior]+1
					except:
						events[log_behavior] = 0
						events[log_behavior] = events[log_behavior]+1
				elif model_behavior != None and model_behavior != ">>":
					try:
						events[model_behavior] = events[model_behavior] + 1
					except:
						events[model_behavior] = 0
						events[model_behavior] = events[model_behavior]+1

	while bool(events):
		popped_event = events.popitem()
		if popped_event[1] > 0:
			unaligned_activities[popped_event[0]] = popped_event[1]

	return unaligned_activities
		
def write_metrics(fitness, precision, generalization, simplicity, event_logs_to_check):

	if event_logs_to_check == "A_only":
		file = open(output_test_metrics_dir + "fitness.txt", "w")
		for idx_type,type in enumerate(fitness):
			for idx_object,object in enumerate(fitness[type]):
				if idx_object < len(fitness[type])-1:
					file.write(type + "," + object + ":" + str(fitness[type][object]) + "\n")
				else:
					file.write(type + "," + object + ":" + str(fitness[type][object]))
			if idx_type < len(fitness)-1:
				file.write("\n")
		file.close()
	else:
		for metric_type in ["fitness", "precision", "generalization", "simplicity"]:
			file = open(output_test_metrics_dir + metric_type + ".txt", "w")

			if metric_type == "fitness":
				for idx_type,type in enumerate(fitness):
					for idx_object,object in enumerate(fitness[type]):
						if idx_object < len(fitness[type])-1:
							file.write(type + "," + object + ":" + str(fitness[type][object]) + "\n")
						else:
							file.write(type + "," + object + ":" + str(fitness[type][object]))
					if idx_type < len(fitness)-1:
						file.write("\n")

			elif metric_type == "precision":
				for idx_object,object in enumerate(precision):
					if idx_object < len(precision)-1:
						file.write(object + ":" + str(precision[object]) + "\n")
					else:
						file.write(object + ":" + str(precision[object]))

			elif metric_type == "generalization":
				for idx_object,object in enumerate(generalization):
					if idx_object < len(generalization)-1:
						file.write(object + ":" + str(generalization[object]) + "\n")
					else:
						file.write(object + ":" + str(generalization[object]))

			elif metric_type == "simplicity":
				for idx_object,object in enumerate(simplicity):
					if idx_object < len(simplicity)-1:
						file.write(object + ":" + str(simplicity[object]) + "\n")
					else:
						file.write(object + ":" + str(simplicity[object]))

			file.close()
	
	return None	
	
def compute_metrics(event_logs, petri_nets, cc_variant, cc_variant_type):

	fitness_values = []

	metrics = {}
	metrics_times = []
	
	for type in event_logs:
		for event_log in event_logs[type]:
			for trace in event_logs[type][event_log]:
				temp = {}
				temp[type] = {}
				temp[type][event_log] = {}
				temp[type][event_log] = (pm4py.objects.log.obj.EventLog)([trace])
				metrics_time = time.time()
				fitness, temp = compute_fitness(temp, petri_nets["N"], cc_variant, cc_variant_type)
				metrics_time = time.time() - metrics_time
				metrics_times.append(metrics_time)
				fitness_values.append(fitness[type][event_log])
		metrics["fitness"] = {}
		metrics["fitness"][type] = {}
		metrics["fitness"][type][event_log] = sum(fitness_values)/len(fitness_values) 
		
	
	metrics["precision"] = compute_precision(event_logs["N"], petri_nets["N"])
	metrics["generalization"] = compute_generalization(event_logs["N"], petri_nets["N"])
	metrics["simplicity"] = compute_simplicity(petri_nets["N"])
	
	metrics_mean_time = sum(metrics_times)/len(metrics_times)
	metrics_std_time = np.std(metrics_times)

	return metrics, metrics_mean_time, metrics_std_time

def write_diagnoses(diagnoses, mode):

	if mode == "Training":
		diagnoses.to_csv(output_training_diagnoses_dir + "diagnoses.csv", index=False)
		
	elif mode == "Test":
		diagnoses.to_csv(output_test_diagnoses_dir + "diagnoses.csv", index=False)

	return None

def write_metrics_timing(metrics_mean_time, metrics_std_time):

	file = open(output_test_metrics_dir + "time.txt", "w")
	file.write("Mean time: " + str(metrics_mean_time) + "\n")
	file.write("Std time: " + str(metrics_std_time))
	file.close()
	
	return None

def write_diagnoses_timing(mean_diagnoses_time, std_diagnoses_time, diagnoses_time_percentage, diagnoses_std_time_percentage):

	file = open(output_test_metrics_dir + "time.txt", "w")
	file.write("Mean time: " + str(mean_diagnoses_time) + "\n")
	file.write("Std time: " + str(std_diagnoses_time) + "\n")
	file.write("Mean time percentage: " + str(diagnoses_time_percentage) + "\n")
	file.write("Std time percentage: " + str(diagnoses_std_time_percentage))
	file.close()

	return None

try:
	mode = sys.argv[1]
	cc_variant = sys.argv[2]
	cc_variant_type = sys.argv[3]
	if mode == "Test":
		event_logs_to_check = sys.argv[4]
		experimentation_type = sys.argv[5]
	elif mode == "Training":
		event_logs_to_check = "All"
except IndexError:
	print("Enter the right number of input arguments.")
	sys.exit()


if mode == "Training":
	event_logs = read_event_logs(mode, event_logs_to_check)
	petri_nets = read_petri_nets(mode)
	diagnoses, ignore, ignore, ignore, ignore = build_diagnoses(event_logs, petri_nets["N"], cc_variant, cc_variant_type)
	write_diagnoses(diagnoses, mode)

elif mode == "Test":
	event_logs = read_event_logs(mode, event_logs_to_check)
	petri_nets = read_petri_nets(mode)
	if event_logs_to_check == "N_only":
		try:				
			metrics, metrics_mean_time, metrics_std_time = func_timeout.func_timeout(timeout=120, func=compute_metrics, args=[event_logs, petri_nets, cc_variant, cc_variant_type])
			write_metrics(metrics["fitness"], metrics["precision"], metrics["generalization"], metrics["simplicity"], event_logs_to_check)
			write_metrics_timing(metrics_mean_time, metrics_std_time )
		except func_timeout.FunctionTimedOut:
			file = open(output_test_metrics_dir + "fitness.txt", "w")
			file.write("-1")
			file.close()
			file = open(output_test_metrics_dir + "precision.txt", "w")
			file.write("-1")
			file.close()
			file = open(output_test_metrics_dir + "generalization.txt", "w")
			file.write("-1")
			file.close()
			file = open(output_test_metrics_dir + "simplicity.txt", "w")
			file.write("-1")
			file.close()
			file = open(output_test_metrics_dir + "time.txt", "w")
			file.write("-1")
			file.close()
		except:
			file = open(output_test_metrics_dir + "fitness.txt", "w")
			file.write("-2")
			file.close()
			file = open(output_test_metrics_dir + "precision.txt", "w")
			file.write("-2")
			file.close()
			file = open(output_test_metrics_dir + "generalization.txt", "w")
			file.write("-2")
			file.close()
			file = open(output_test_metrics_dir + "simplicity.txt", "w")
			file.write("-2")
			file.close()
			file = open(output_test_metrics_dir + "time.txt", "w")
			file.write("-2")
			file.close()
	if experimentation_type == "AnomalyDetection":
		diagnoses, mean_diagnoses_time, std_diagnoses_time, diagnoses_time_percentage, diagnoses_std_time_percentage = build_diagnoses(event_logs, petri_nets["N"], cc_variant, cc_variant_type)
		write_diagnoses(diagnoses, mode)
		write_diagnoses_timing(mean_diagnoses_time, std_diagnoses_time, diagnoses_time_percentage, diagnoses_std_time_percentage)


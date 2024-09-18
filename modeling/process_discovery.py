from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.objects.conversion.process_tree import converter as pt_converter

import pm4py
import sys
import os
import time
#import pm4py.algo.discovery as pd

input_dir = "Input/PD/"
input_training_dir = input_dir + "Training/"
input_training_eventlogs_dir = input_training_dir + "EventLogs/"

output_dir = "Output/PD/"
output_training_dir = output_dir + "Training/"
output_training_petrinets_dir = output_training_dir + "PetriNets/"
output_training_metrics_dir = output_training_dir + "Metrics/"

def read_event_logs():

	event_logs = {}

	for type in os.listdir(input_training_eventlogs_dir):
		event_logs[type] = {}
		for event_log_file in os.listdir(input_training_eventlogs_dir + type + "/"):
			event_logs[type][event_log_file.split(".xes")[0]] = xes_importer.apply(input_training_eventlogs_dir + type + "/" + event_log_file)

	return event_logs
	
def process_discovery(event_logs, pd_variant):

	petri_nets = {}
	parameters = {}

	for type in event_logs:
		petri_nets[type] = {}
		for event_log in event_logs[type]:
			petri_nets[type][event_log] = {}

			tm = 0
			tm = time.time()

			# The following code block works with pm4py 2.2.x
			'''
			if pd_variant == "im":
				parameters[pd.inductive.algorithm.Variants.IMf.value.Parameters.NOISE_THRESHOLD] = 0.0
				petri_nets[type][event_log]["network"], petri_nets[type][event_log]["initial_marking"], petri_nets[type][event_log]["final_marking"] = pd.inductive.algorithm.apply(event_logs[type][event_log], variant=pd.inductive.algorithm.Variants.IMf, parameters=parameters)
			elif pd_variant == "imf_20":
				parameters[pd.inductive.algorithm.Variants.IMf.value.Parameters.NOISE_THRESHOLD] = 0.2
				petri_nets[type][event_log]["network"], petri_nets[type][event_log]["initial_marking"], petri_nets[type][event_log]["final_marking"] = pd.inductive.algorithm.apply(event_logs[type][event_log], variant=pd.inductive.algorithm.Variants.IMf, parameters=parameters)
			elif pd_variant == "imd":
				petri_nets[type][event_log]["network"], petri_nets[type][event_log]["initial_marking"], petri_nets[type][event_log]["final_marking"] = pd.inductive.algorithm.apply(event_logs[type][event_log], variant=pd.inductive.algorithm.Variants.IMd, parameters=parameters)
			elif pd_variant == "alpha_plus":
				petri_nets[type][event_log]["network"], petri_nets[type][event_log]["initial_marking"], petri_nets[type][event_log]["final_marking"] = pd.alpha.algorithm.apply(event_logs[type][event_log], variant=pd.alpha.algorithm.Variants.ALPHA_VERSION_PLUS, parameters=parameters)
			elif pd_variant == "heuristics":
				petri_nets[type][event_log]["network"], petri_nets[type][event_log]["initial_marking"], petri_nets[type][event_log]["final_marking"] = pd.heuristics.algorithm.apply(event_logs[type][event_log], variant=pd.heuristics.algorithm.Variants.CLASSIC, parameters=parameters)
			'''
				
			if pd_variant == "im":
				petri_nets[type][event_log]["network"], petri_nets[type][event_log]["initial_marking"], petri_nets[type][event_log]["final_marking"] = pm4py.discover_petri_net_inductive(event_logs[type][event_log])
			elif pd_variant == "imf_20":
				petri_nets[type][event_log]["network"], petri_nets[type][event_log]["initial_marking"], petri_nets[type][event_log]["final_marking"] = pm4py.discover_petri_net_inductive(event_logs[type][event_log], noise_threshold = 0.2)
			elif pd_variant == "imd":
				dfg = pm4py.discover_dfg_typed(pm4py.convert_to_dataframe(event_logs[type][event_log]))
				ptree = pm4py.discover_process_tree_inductive(dfg)
				petri_nets[type][event_log]["network"], petri_nets[type][event_log]["initial_marking"], petri_nets[type][event_log]["final_marking"] = pm4py.convert_to_petri_net(ptree)
			elif pd_variant == "alpha":
				petri_nets[type][event_log]["network"], petri_nets[type][event_log]["initial_marking"], petri_nets[type][event_log]["final_marking"] = pm4py.discover_petri_net_alpha(event_logs[type][event_log])
			elif pd_variant == "heuristics":
				petri_nets[type][event_log]["network"], petri_nets[type][event_log]["initial_marking"], petri_nets[type][event_log]["final_marking"] = pm4py.discover_petri_net_heuristics(event_logs[type][event_log])
			elif pd_variant == "ilp":
				petri_nets[type][event_log]["network"], petri_nets[type][event_log]["initial_marking"], petri_nets[type][event_log]["final_marking"] = pm4py.discover_petri_net_ilp(event_logs[type][event_log])	
		
			tm = time.time() - tm

			petri_nets[type][event_log]["places"] = len(petri_nets[type][event_log]["network"].places)
			petri_nets[type][event_log]["transitions"] = len(petri_nets[type][event_log]["network"].transitions)
			petri_nets[type][event_log]["arcs"] = len(petri_nets[type][event_log]["network"].arcs)
			petri_nets[type][event_log]["time"] = tm

	return petri_nets
	
def export_petri_nets(petri_nets):
	for type in petri_nets:
		isExist = os.path.exists(output_training_petrinets_dir + type)
		if not isExist:
			os.makedirs(output_training_petrinets_dir + type)
		for petri_net in petri_nets[type]:
			pnml_exporter.apply(petri_nets[type][petri_net]["network"], petri_nets[type][petri_net]["initial_marking"], output_training_petrinets_dir + type + "/" + petri_net + ".pnml", final_marking = petri_nets[type][petri_net]["final_marking"])	

			petri_nets[type][petri_net]["size"] = os.path.getsize(output_training_petrinets_dir + type + "/" + petri_net + ".pnml")/1000

	return petri_nets

def write_statistics(petri_nets):
	for type in petri_nets:
		for petri_net in petri_nets[type]:
			file = open(output_training_metrics_dir + "statistics.txt", "w")
			file.write("Time: " + str(petri_nets[type][petri_net]["time"]) + "\n")
			file.write("Places: " + str(petri_nets[type][petri_net]["places"]) + "\n")
			file.write("Transitions: " + str(petri_nets[type][petri_net]["transitions"]) + "\n")
			file.write("Arcs: " + str(petri_nets[type][petri_net]["arcs"]) + "\n")
			file.write("Size: " + str(petri_nets[type][petri_net]["size"]))
			file.close()

try:
	pd_variant = sys.argv[1]
except IndexError:
	print("Enter the right number of input arguments.")
	sys.exit()
	
event_logs = read_event_logs()
petri_nets = process_discovery(event_logs, pd_variant)
petri_nets = export_petri_nets(petri_nets)
write_statistics(petri_nets)

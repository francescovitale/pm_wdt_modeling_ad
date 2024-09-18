# Requirements and instructions to run the framework

Before delving into the details of the project files, please consider that this project has been executed on a Windows 10 machine with Python 3.11.5. There are a few libraries that have been used within Python modules. Among these, there are:

- scipy 1.11.2
- scikit-learn 1.3.0
- pm4py 2.7.11.11

Please note that the list above is not comprehensive and there could be other requirements for running the project.

## Modeling

The modeling part of the framework is contained in the modeling folder. The modeling part runs by executing the DOS experimentation_modeling.bat script. This script includes a set of parameters that can be customized to set: the split fraction (i.e., quantization level) with which case-study data is preprocessed, the process discovery variant to use, the conformance checking variant to use, and the number of repetitions with which a technique of the framework is executed.

The experimentation_modeling.bat script first navigates the WDT_DATA folder and executes a DOS script to extract the cycles from the Water Distribution Testbed (WDT) data. Then, it combines the execution of the event_extraction.py, process_discovery.py and conformance_checking.py Python scripts, which, respectively, extract the training and test event logs from the WDT cycles, build a Petri net based on the training event logs, and evaluate the four metrics to assess the Petri net quality. The results are stored under the Results folder.

## Anomaly detection

The anomaly detection part of the framework is contained in the anomaly_detection folder. The anomaly detection part is split into two experimental testbeds, under the pn_quality_impact and ilp_assessment folders. The first testbed evaluates the impact of the quality of the Petri net on anomaly detection and time performance, using a set of machine learning algorithms. This testbed requires executing two Python scripts, namely ml_val.py to first compute the classifiers' hyperparameters and ml_test.py to compute the results. The second testbed evaluates anomaly detection and time performance of the Integer Linear Programming-based miner, which has shown the best modeling results during the experimentation. The results regard the evolution of detection effectiveness and time performance as the quantization level decreases, according to different classifiers. This testbed requires executing three Python scripts, namely ml_val.py to first compute the classifiers' hyperparameters, ml_test.py to compute the results, and plot.py to view the results.

# How to generate results

## Network Fitting

There are several experiments. For each experiment in the list the use the steps below with the appropriate methods:
- Full planning ablation study: investigates the impact of each of the components of the full planning state vector.
- Trajectory tracking number of waypoints: evaluates the number of waypoints required to best fit the data
- End-to-end number of beams: evaluates the number of beams required by the end-to-end agent
- End-to-end speed communication: investigates different methods of communicating speed to the agent
- Comparison of full planning, trajectory tracking and end-to-end: compares the three approaches


1. Generate data: 
    - Use the `NetworkFittingData.yaml` config file to adjust parameters
    - Run the `GenerateDataSet.py` file to generate the dataset
2. Create formatted datasets to train networks
    - Use `Build_endToEnd_datasets.py` to create the end-to-end data sets
    - Use `Build_planning_datasets.py` to create the planning data sets
3. Train networks
    - Use `TrainNetworks_SingleLoss.py` to train the networks and save the training and testing losses as the sum of the steering and speed losses.
    - Use `TrainNetworks_SeperateLosses.py` to train the networks and save the training and testing losses as the *individual* steering and speed losses.
4. Plot the results:
    - Full planning ablation study: barplot of speed and steering losses
    - Trajectory tracking number of waypoints: graph of training and test losses for different numbers of waypoints
    - End-to-end number of beams: graph of training and test losses for different numbers of beams
    - End-to-end speed communication: barplot of speed and steering losses
    - Comparison of full planning, trajectory tracking and end-to-end: graph of training and test losses and table of rescaled values


### Deep reinforcement learning experiments

Agents are trained and tested and the data is analysed.
The process for performing these experiments is to train and test the agents, analyse the data and then plot the data.


### Training Agents

The experiments train agents on the MCO and GBR tracks.
For the paper, five repeats are performed on the MCO track and 3 repeats on the GBR track. This is changed by commenting/uncommenting the respective lines in the `FinalExperiment.yaml` file.
- The `run_experiments.py` file can be run to train and test the agents. It will read from the yaml confif file and do the experiments listed in there.
- When the agents are trained, they will be automatically tested on the training map. They must be manually tested on the other maps using the `run_general_test_batch` function
- The classical agents must be tested without training them using the `run_testing_batch` function

**Analysing Training Data:**

Four graphs are used to analyse the training performance:
1. TrainingReward
2. TrainingProgress
3. TrainingCrashes
4. TrainingLapTimes
All the files contain a simple script to generate the appropriate graph

The two 'fast' scripts are there for debugging purposes.

### Analysing Test Data

During testing, the state of the agent and the action selected at each timestep is recorded.

**Statistics:**

The first analysis is to calculate the statistics for the training data.

- `CaclulateMainStatistics.py`: Calculates the success rate, lap time and average progress for each testing run.
- `CaclulateDetailStatistics.py`: Calculates the time, progress, total distance, average and standard deviation of velocity and curvature, Q1, Q2, Q3 of lateral and speed deviation.
- `MakeRepetitionSummary.py`: makes a summary of the performance of the repetitions.
- `MakeRepetitionBarPlot.py`: makes a bar plot of the training repetitions on the MCO map
- `MakePerformanceTable.py`: makes a table of the lap times and progresses of the last repetition of the agents trained and tested on the MCO track.
- `DeviationBarPlot.py`: plots the speed and lateral deviations experience by the agents compared to the racing line.





import pandas as pd
import glob







def build_experiment_df(path):
    vehicle_folders = glob.glob(f"{path}*/")
    print(f"{len(vehicle_folders)} folders found")

    experiment_data = []

    for j, agent_path in enumerate(vehicle_folders):
        vehicle_name = agent_path.split("/")[-2]
        if vehicle_name == "Imgs": continue

        algorithm = vehicle_name.split("_")[1]
        architecture = vehicle_name.split("_")[2]
        train_map = vehicle_name.split("_")[3].upper()
        train_id = vehicle_name.split("_")[4]
        repetition = vehicle_name.split("_")[7]

        try:
            df = pd.read_csv(agent_path + "MainAgentData.csv")
        except Exception as e: 
            print(f"{e} for {vehicle_name}")
            continue

        for test_map in df.TestMap.unique():

            test_df = df[df.TestMap == test_map]
            test_df = test_df.drop(["Lap", "TestMap"], axis=1)

            agent_data = {"Algorithm": algorithm, "Architecture": architecture, "TrainMap": train_map, "TrainID": train_id, "Repetition": repetition}
            agent_data["TestMap"] = test_map.upper()
            agent_data.update(test_df.mean().to_dict())

            experiment_data.append(agent_data)

    experiment_df = pd.DataFrame(experiment_data)
    experiment_df.to_csv(path + "ExperimentData.csv", index=False)



def main():
    experiment_name = "FinalExperiment"
    set_n = 1
    experiment_path = f"Data/{experiment_name}_{set_n}/"

    build_experiment_df(experiment_path)



if __name__ == '__main__':
    main()



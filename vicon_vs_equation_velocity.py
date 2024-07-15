import csv
from os import listdir
from os.path import isfile, join
import numpy as np
from helper import *

# Load the ground truth velocities from the vicon_estimated_velocities.csv file
ground_truth_file = "vicon_estimated_velocities.csv"
ground_truth_dict = {}

with open(ground_truth_file, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        ground_truth_dict[row['PWM']] = float(row['estimated_velocity'])

data_folder = "datasets"
bin_files = [f for f in listdir(data_folder) if isfile(join(data_folder, f)) and f.endswith('.bin') and not f.startswith('only_sensor')]

results = []
total_squared_error = 0
total_files = 0

for file_name in bin_files:
    print("Processing file ", file_name)
    info_dict = get_info(file_name)
    pwm_value = int(info_dict[' PWM'][0])
    run_data_read_only_sensor(info_dict)
    bin_filename = 'datasets/only_sensor' + info_dict['filename'][0]
    bin_reader = RawDataReader(bin_filename)
    total_frame_number = int(info_dict[' Nf'][0])
    pointCloudProcessCFG = PointCloudProcessCFG()
    print_info(info_dict)

    all_range_index = []
    all_consistent_peaks = []
    consistent = True
    velocity_array = []

    # Iterate through each frame in the current file
    for frame_no in range(total_frame_number):
        bin_frame = bin_reader.getNextFrame(pointCloudProcessCFG.frameConfig)
        np_frame = bin2np_frame(bin_frame)
        frameConfig = pointCloudProcessCFG.frameConfig
        reshapedFrame = frameReshape(np_frame, frameConfig)
        rangeResult = rangeFFT(reshapedFrame, frameConfig)
        
        range_result_absnormal_split = []
        for i in range(pointCloudProcessCFG.frameConfig.numTxAntennas):
            for j in range(pointCloudProcessCFG.frameConfig.numRxAntennas):
                r_r = np.abs(rangeResult[i][j])
                r_r[:,0:10] = 0
                min_val = np.min(r_r)
                max_val = np.max(r_r)
                r_r_normalise = (r_r - min_val) / (max_val - min_val) * 1000
                range_result_absnormal_split.append(r_r_normalise)
        range_abs_combined_nparray = np.zeros((pointCloudProcessCFG.frameConfig.numLoopsPerFrame, pointCloudProcessCFG.frameConfig.numADCSamples))
        for ele in range_result_absnormal_split:
            range_abs_combined_nparray += ele
        range_abs_combined_nparray /= (pointCloudProcessCFG.frameConfig.numTxAntennas * pointCloudProcessCFG.frameConfig.numRxAntennas)
        range_abs_combined_nparray_collapsed = np.sum(range_abs_combined_nparray, axis=0) / pointCloudProcessCFG.frameConfig.numLoopsPerFrame
        peaks, _ = find_peaks(range_abs_combined_nparray_collapsed)
        intensities_peaks = [[range_abs_combined_nparray_collapsed[idx],idx] for idx in peaks]
        peaks = [i[1] for i in sorted(intensities_peaks, reverse=True)[:3]]
        all_range_index.append(peaks)
        # For first frame take all peaks as consistent peaks
        if frame_no == 0:
            all_consistent_peaks.append(peaks)
        else:
            previous_peaks = all_range_index[frame_no-1]
            current_peaks = all_range_index[frame_no]
            consistent_peaks = get_consistent_peaks(previous_peaks, current_peaks, threshold=10)
            all_consistent_peaks.append(consistent_peaks)
        vel_array_frame = np.array(get_velocity(rangeResult,all_consistent_peaks[frame_no],info_dict)).flatten()
        mean_velocity = (vel_array_frame.mean())
        velocity_array.append(mean_velocity)
    mean_velocity = np.mean(velocity_array)
    estimated_velocity = ground_truth_dict.get(str(pwm_value), -1)
    if(estimated_velocity==-1):
        continue
    speed_difference = mean_velocity - estimated_velocity*0.01
    results.append({
        'file_name': file_name,
        'mean_velocity': mean_velocity,
        'estimated_velocity': estimated_velocity,
        'speed_difference': speed_difference
    })
    total_squared_error += speed_difference ** 2
    total_files += 1

# Calculate the Mean Squared Error
mse = total_squared_error / total_files if total_files else 0

# Save the results to a new CSV file
output_file = "velocity_comparison_results.csv"
with open(output_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['file_name', 'mean_velocity', 'estimated_velocity', 'speed_difference'])
    writer.writeheader()
    writer.writerows(results)

print(f"Mean Squared Error: {mse}")

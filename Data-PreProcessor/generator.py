import pandas as pd
import numpy as np
from lib.sampler import WindowSampler
from columns import columns

labels = pd.read_csv("labels.txt", delim_whitespace=True, names=["Exp","Subject","Activity","Start","End"])
# accData = pd.read_csv("acc_exp01_user01.txt", delim_whitespace=True, names=["AccX","AccY","AccZ"])
# gyroData = pd.read_csv("gyro_exp01_user01.txt", delim_whitespace=True, names=["GyroX","GyroY","GyroZ"])
# final_data = accData.join(gyroData)
# print(final_data.loc[0:10])
activities = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

# Generate final table at once
# label_sample = {
#     "Subject": [labels["Subject"][0] for x in range(labels["Start"][0],labels["End"][0]+1)],
#     "Activity": [activities[labels["Activity"][0]] for x in range(labels["Start"][0],labels["End"][0]+1)]
# }
# frame_sample = pd.DataFrame(data=label_sample,index=range(labels["Start"][0],labels["End"][0]+1))
# print(frame_sample.head())
# print(frame_sample.shape)
# final_table = final_data.loc[labels["Start"][0]:labels["End"][0]].join(frame_sample)
# print(final_table.head())
# print(final_table.tail())
# for index, row in labels.iterrows():
#     print(row["Exp"])

# TODO generate tables and send it for sampling process
# name = "acc_exp" + format(1,'02d') + "_user" + format(1,'02d') + ".txt"
# print(name)
exp_no = 1
u_no = 1
accData = pd.read_csv("acc_exp01_user01.txt", delim_whitespace=True, names=["AccX","AccY","AccZ"])
gyroData = pd.read_csv("gyro_exp01_user01.txt", delim_whitespace=True, names=["GyroX","GyroY","GyroZ"])
final_data = accData.join(gyroData)
exp_no = accData.size
myDataset = pd.DataFrame()

for index, row in labels.iterrows():

    if row["Exp"] != exp_no:
        exp_no = row["Exp"]
        u_no = row["Subject"]
        name = "acc_exp" + format(exp_no,'02d') + "_user" + format(u_no,'02d') + ".txt"
        accData = pd.read_csv(name,delim_whitespace=True,names=["AccX","AccY","AccZ"])
        name = "gyro_exp" + format(exp_no,'02d') + "_user" + format(u_no,'02d') + ".txt"
        gyroData = pd.read_csv(name, delim_whitespace=True, names=["GyroX","GyroY","GyroZ"])
        final_data = accData.join(gyroData)

        if row["Activity"] < 7:
            sampler = WindowSampler(final_data.loc[labels["Start"][index]:labels["End"][index]],
            128, 64, labels["Start"][index], labels["End"][index])
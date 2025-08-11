import os
import shutil
import json
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
import seaborn as sns

colors_table = {
    "tableA": sns.color_palette("pastel")[0],
    "tableB": sns.color_palette("pastel")[5],
    "tableC": sns.color_palette("pastel")[2],
    "tableD": sns.color_palette("pastel")[3],
    "tableE": sns.color_palette("pastel")[4]
}

def time_hhmmss_to_seconds(time_hhmmss):
    return int(time_hhmmss[:2]) * 3600 + int(time_hhmmss[2:4]) * 60 + int(time_hhmmss[4:6])

def time_seconds_to_hhmmss(time_seconds, return_human_readable=False):
    time_seconds = int(time_seconds)
    if return_human_readable:
        return f"{time_seconds//3600:02d}:{(time_seconds%3600)//60:02d}:{int((time_seconds%3600)%60):02d}"
    else:
        return f"{time_seconds//3600:02d}{(time_seconds%3600)//60:02d}{int((time_seconds%3600)%60):02d}"

TIME_START_HHMMSS = "150000"
TIME_END_HHMMSS = "180000"
TIME_START_SECONDS = time_hhmmss_to_seconds(TIME_START_HHMMSS)
TIME_END_SECONDS = time_hhmmss_to_seconds(TIME_END_HHMMSS)

def lowercase_table_name(table_name):
    return table_name.replace("Table", "table").replace(" ", "")

def proc_labels(frame_labels_list):
    def label_name_mods(ln):
        ln = ln.split("_")[0]
        if '|' in ln:
            ln = ln.split('|')[-1]
        return ln
    categories = sorted(list(set([label_name_mods(lab["category"]) for lab in frame_labels_list])))
    return categories


def process_frames(data_raw, start_time_hhmmss):
    start_time_seconds = time_hhmmss_to_seconds(start_time_hhmmss)
    timestamped_scene_graphs = {}
    for frame in data_raw:
        timestamp_relative = int(frame["name"].split("_")[-1].split(".")[0])//1000
        timestamp_absolute = start_time_seconds + timestamp_relative
        if timestamp_absolute > TIME_START_SECONDS and timestamp_absolute < TIME_END_SECONDS:
            labels_list = proc_labels(frame["labels"])
            timestamped_scene_graphs[timestamp_relative] = labels_list
            if "person" in timestamped_scene_graphs[timestamp_relative]:
                timestamped_scene_graphs[timestamp_relative].remove("person")
                timestamped_scene_graphs[timestamp_relative] += ["person"]*labels_list.count("person")
    return timestamped_scene_graphs


def embed_object(object_name, object_masterlist):
    return object_masterlist.index(object_name)
from config import openml_data_path
import glob, json

def construct_task_list():
    ret = {}
    for path in glob.glob(f"{openml_data_path}/*/head.json"):
        segs = path.split("/")
        task_name = segs[-2]
        with open(path, "r") as fd:
            header_info = json.load(fd)
        if "text header" in header_info and header_info["text header"] == "false":
            continue

        task_id = header_info["task_id"]
        ret[task_name] = task_id
    
    return ret


TASK_ID = construct_task_list()
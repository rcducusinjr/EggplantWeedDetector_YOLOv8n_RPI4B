'''
Insert your API key from Roboflow to the placeholder.

If you use this dataset in a research paper, please cite it using the following BibTeX:

@misc{
    eggplant-and-weeds-detection_dataset,
    title = { Eggplant and Weeds Detection Dataset },
    type = { Open Source Dataset },
    author = { UPLB Ducusin SP },
    howpublished = { /url{ https://universe.roboflow.com/uplb-ducusin-sp/eggplant-and-weeds-detection } },
    url = { https://universe.roboflow.com/uplb-ducusin-sp/eggplant-and-weeds-detection },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2024 },
    month = { apr },
    note = { visited on 2024-04-05 },
}
'''

from roboflow import Roboflow

rf = Roboflow(api_key="610ndPHwIUHkaPqpdvbj")
project = rf.workspace("uplbducusinsp-us63c").project("eggplant-and-weeds-detection")
version = project.version(10)
dataset = version.download("yolov8")
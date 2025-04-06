import gdown

url = "https://drive.google.com/file/d/1z0mreUtRmR-P-magILsDR3T7M6IkGXtY/view?usp=sharing"

url_list = url.split("/")
print(url_list)

file_id = url.split("/")[-2]
print(file_id)

prefix = "https://drive.google.com/uc?/export=download&id="
gdown.download(prefix+file_id, "Chest-CT-Scan-data.zip")

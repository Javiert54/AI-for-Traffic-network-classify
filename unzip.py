import os
import zipfile
def unzip(path:str, target:str) -> None:
    """
    Unzip a zip file to a target directory.
    
    Args:
        path (str): The path to the zip file.
        target (str): The target directory to unzip the files into.
    """
    import zipfile
    import os

    # Create the target directory if it doesn't exist
    os.makedirs(target, exist_ok=True)

    # Unzip the file
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(target)

unzip("datasets/02-20-2018.zip", "datasets/")
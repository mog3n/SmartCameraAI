import os
from glob import glob
import time
import io
import json
import numpy as np
import face_recognition
import shutil
import logging

logging.basicConfig(level=logging.INFO)

def load_encodings():
    file_directory = os.path.join(os.getcwd(), 'faces', 'faces.json')
    file = io.open(file_directory, 'r')
    face_obj = json.loads(file.read())
    file.close()
    return face_obj


def save_encodings(encodings):
    file_directory = os.path.join(os.getcwd(), 'faces', 'faces.json')
    file = io.open(file_directory, 'w')
    file.write(json.dumps(encodings))
    file.close()
    # logging.info("saved encodings to file")


class Groups:
    def __init__(self, groups={}, index=0):
        self.groups = groups
        self.index = index

    # Checks if the key is in any of the groups. Returns which group
    def get_group_name(self, key):

        for group_name in self.groups.keys():
            group = self.groups[group_name]
            if key in group:
                return group_name

        # Key does not belong in any group
        return None

    def add_to_group(self, group_name, key):
        self.groups[group_name].append(key)

    def create_new_group_with_key(self, key):
        group_name = "group" + str(self.index)
        self.groups[group_name] = [key]
        self.index += 1
        return group_name

    def get_groups(self):
        return self.groups

    def is_empty(self):
        return self.groups == {}


def save_groups(groups):
    data_to_save = {
        'groups': groups.get_groups(),
        'index': groups.index
    }
    file_path = os.path.join(os.getcwd(), 'faces', 'groups.json')
    fs = open(file_path, 'w')
    fs.write(json.dumps(data_to_save))
    fs.close()


def load_groups():
    file_path = os.path.join(os.getcwd(), 'faces', 'groups.json')
    fs = open(file_path, 'r')
    dict = json.loads(fs.read())
    fs.close()
    gr = Groups(dict['groups'], dict['index'])
    return gr


def group_faces():
    encodings = load_encodings()
    groups = Groups()

    faces = list(encodings.keys())
    # While there are faces waiting to be put into a group
    while faces:
        face = faces.pop()

        # if this is the first face, create a new group
        if groups.is_empty():
            groups.create_new_group_with_key(face)
            continue

        # Get the group dictionary
        groups_dict = groups.get_groups()

        # Stores whether or not the face was added to a group
        face_added_to_group = False

        # Iterate through each group and compare the current face with the first face in the group
        for group_name, group_members in groups_dict.items():

            consensus = 0
            n = len(group_members)

            # compare face with every group member
            for member in group_members:

                # Get the encodings of the current face and the member face
                current_face_encoding = encodings[face]['encodings'][0]  # Take the first encoding
                member_face_encoding = encodings[member]['encodings'][0]

                # Convert faces to ndarray
                current_face_encoding = np.ndarray((1, 128), buffer=np.array(current_face_encoding))
                member_face_encoding = np.ndarray((1, 128), buffer=np.array(member_face_encoding))

                # Compare encoding
                result = face_recognition.compare_faces(current_face_encoding, member_face_encoding, tolerance=0.5)

                if result[0]:
                    # Matched with face. Added +1 to consensus
                    consensus += 1

            percentage_matched = consensus/n
            threshold = 0.7  # minimum percentage of matches to allow into the group

            if percentage_matched > threshold:
                # Add faces to the group
                logging.info("adding to group", group_name, face)
                groups.add_to_group(group_name, face)
                face_added_to_group = True
                break

        # Create a new group if the face was not added to the group
        if not face_added_to_group:
            new_group_name = groups.create_new_group_with_key(face)
            logging.info("making new group", new_group_name, face)

    # Make groups
    groups_dict = groups.get_groups()
    for group_name, members in groups_dict.items():

        # Create group directory
        path = os.path.join(os.getcwd(), "faces", group_name)
        if not os.path.exists(path):
            os.mkdir(path)

        # Copy member group
        for member in members:
            old_image_path = os.path.join(os.getcwd(), 'faces', 'temp', member)
            new_image_path = os.path.join(os.getcwd(), "faces", group_name, member)
            shutil.copyfile(old_image_path, new_image_path)

    # Save the group file
    save_groups(groups)
    logging.info("Done")


def load_analyzed_files():
    path = os.path.join(os.getcwd(), 'faces', 'analyzed_files.json')
    if os.path.exists(path):
        file = open(path, 'r')
        db = json.loads(file.read())

        db['face_num'] = len(db['files'])  # just in case
        file.close()
    else:
        db = {
            'files': [],
            'face_num': 0
        }
        file = open(path, 'w')
        file.write(json.dumps(db))

    return db


def save_analyzed_files(db):
    path = os.path.join(os.getcwd(), 'faces', 'analyzed_files.json')
    file = open(path, 'w')
    file.write(json.dumps(db))
    file.close()
    # logging.info("List of files analyzed saved.")


def extract_encodings():

    db = load_analyzed_files()

    encodings = {}
    people = glob(os.path.join(os.getcwd(), 'detected_objects', 'person', '*.jpg'))

    for person_image_path in people:

        file_name = os.path.basename(person_image_path)  # get filename instead of entire path

        # Check if the file has been already seen
        if file_name in db['files']:
            # Skip this file
            # logging.info("Skipping already analyzed file")
            continue

        # Start counting up the face # seen
        db['face_num'] += 1
        image = face_recognition.load_image_file(person_image_path)
        face_locations = face_recognition.face_locations(image)

        # Get file name
        file_name = os.path.basename(person_image_path)

        # Check if thre are any faces in the image
        if face_locations != []:

            location_to_copy_to = os.path.join(os.getcwd(), 'faces', 'temp', file_name)
            # Save copy file to faces
            shutil.copyfile(person_image_path, location_to_copy_to)

            # Get face encodings for this image
            ndarray_face_encodings = face_recognition.face_encodings(image)
            list_face_encodings = []
            # convert each encoding to a list
            for encoding in ndarray_face_encodings:
                list_face_encodings.append(encoding.tolist())

            # Save locations and encodings
            data_to_save = {
                'locations': face_locations,
                'encodings': list_face_encodings
            }

            encodings[file_name] = data_to_save  # Convert ndarray to list
            logging.info("added a face encoding to " + file_name)
            save_encodings(encodings)

        db['files'].append(file_name)
        save_analyzed_files(db)

    save_analyzed_files()
    save_encodings(encodings)
    logging.info("done")


if __name__ == "__main__":
    if not os.path.exists(os.path.join(os.getcwd(), 'faces', 'temp')):
        os.mkdir(os.path.join(os.getcwd(), 'faces', 'temp'))

    while True:
        extract_encodings()
        # group_faces()
        time.sleep(5)

import numpy as np
import requests
import json
import cv2
import argparse
import os

address = 'http://127.0.0.1:5000'
img_path = 'images/girl_image.jpg'
save_dir = "outputs/"

def parse_args():
    parser = argparse.ArgumentParser(description="This helps to send Image Data")

    parser.add_argument("-a", "--address", type=str, default=address, help="endpoint you want to hit")
    parser.add_argument("-i", "--image_path", type=str, required=True, help="path to the input image")
    parser.add_argument("-o", "--output_dir", type=str, default="outputs/", help="path where the output image is stores")

    args = parser.parse_args()
    return args

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

def post_image(img_path, URL):
    """ post image and return the response """
 
    img = open(img_path, 'rb').read()
    response = requests.post(URL, data=img, headers=headers)
    return response

def parse_response(response):
    response = json.loads(response.text)
    detections = response["detections"]
    detected_people = []
    person_count = 0
    for item in detections:
        for key in item:
            if key == "person" and item[key] >= '90%':
                detected_people.append(item)
                person_count +=1
    return response, detected_people, person_count 

def write_image(save_dir, parsed_response):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print("the image is being written in {}".format(save_dir))
    img_array = np.asarray(parsed_response["image data"])
    cv2.imwrite(save_dir + "server_output.jpg", img_array)
    return img_array

if __name__ == "__main__":
    args = parse_args()
    url = args.address + '/object_detection'
    response = post_image(args.image_path, url)
    parsed_response, detected_people, people_count = parse_response(response)

    print("number of people detected:", people_count)
    print("detections are:", detected_people)
    print("the image size is:", parsed_response["image size"])

    write_image(args.output_dir, parsed_response)
    
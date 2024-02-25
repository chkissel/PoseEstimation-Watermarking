import cv2
import math
import statistics
import hashlib

print("---- START ----")

# Specify the paths for the 2 files 
protoFile = "/path/to/models/pose/body_25/pose_deploy.prototxt"
weightsFile = "/path/to/models/pose/body_25/pose_iter_584000.caffemodel"

img_path = "/path/to/image.jpeg"

print("Initialising Deep Neural Network.")

# Read the network into Memory 
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile) 

# Read image 
frame = cv2.imread(img_path)   

# Specify the input image dimensions 
inWidth = frame.shape[1]  
inHeight = frame.shape[0]  
  
# Prepare the frame to be fed to the network 
inpBlob = cv2.dnn.blobFromImage( 
    frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False) 
  
# Set the prepared object as the input blob of the network 
net.setInput(inpBlob) 

output = net.forward() 

H = output.shape[2] 
W = output.shape[3] 

# Empty list to store the detected keypoints 
points = [] 
number_relevant_keypoints = 15
threshold = 0.1

for i in range(number_relevant_keypoints): 
    
    # Confidence map of corresponding body's part. 
    probMap = output[0, i, :, :]
    
    # Find global maxima of the probMap. 
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap) 
  
    # Scale the point to fit on the original image 
    x = (inWidth * point[0]) / W 
    y = (inHeight * point[1]) / H 

    if prob > threshold: 
       
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), 
                thickness=-1, lineType=cv2.FILLED) 
        """
        cv2.putText(frame, "{}".format(i), (int(x), int( 
            y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA) 
        """

        # Add the point to the list if the probability is greater than the threshold 
        points.append((int(x), int(y))) 
    else: 
        points.append(None) 
    
joint_pairs = [(0,1), (1,8), (8,12), (8,9), (9,10), (10,11), (12,13), (2,5), (2,3), (3,4), (5,6), (6,7), (13,14)]

for pair in joint_pairs: 
    partA = pair[0] 
    partB = pair[1] 
    
    if points[partA] and points[partB]: 
        cv2.line(frame, points[partA], points[partB], (0, 255, 0), 2) 

S = 0.5
print(f"Scale factor = {S}")

# Read again the original image without the drawing
org = cv2.imread(img_path)
scaled = cv2.resize(org, (0,0), fx=S, fy=S) 

inWidth = scaled.shape[1]  
inHeight = scaled.shape[0]  

# Set up network input just as before 
inpBlob = cv2.dnn.blobFromImage( 
    scaled, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False) 

net.setInput(inpBlob) 
output = net.forward() 

H = output.shape[2] 
W = output.shape[3] 

# Empty list to store the detected keypoints 
scaled_points = [] 

for i in range(number_relevant_keypoints): 
    
    # confidence map of corresponding body's part. 
    probMap = output[0, i, :, :]
    
    # Find global maxima of the probMap. 
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap) 
  
    # Scale the point to fit on the original image 
    x = (inWidth * point[0]) / W 
    y = (inHeight * point[1]) / H 
    
    if prob > threshold: 
        scaled_points.append((int(x), int(y))) 
    else: 
        scaled_points.append(None) 

print("---- OBSERVATIONS ----")

"""
observation #1: 
Will the key joints remain in the proportionally same places after scaling the image?
"""

scale_factor = 1/S
deviations = []
for i, point in enumerate(scaled_points):
    try:
        # scale the points in relation to the original image size
        x = scale_factor * point[0] 
        y = scale_factor * point[1]

        deviations.append(math.dist(points[i], (x,y)))
    except:
        pass

print(f"Average deviation of key joint locations after scaling: {round(statistics.mean(deviations), 4)}")

"""
observation #2: 
Will the distance between two key joints remain the ame after scaling the image?    
"""

deviations = []
for pair in joint_pairs:
    if point != None:
        point_a = pair[0] 
        point_b = pair[1]  
    try:
        distance = (math.dist(points[point_a], points[point_b])) - (math.dist((scaled_points[point_a]), (scaled_points[point_b])) * scale_factor)
        deviations.append(abs(distance))
    except:
        pass

print(f"Average deviation of distances between key joint pairs after scaling: {round(statistics.mean(deviations), 4)}")

"""
observation #3: 
Will the ratio of two distances between the key joints remain the ame after scaling the image?    
"""

deviations = []
for i in range(len(joint_pairs)-1):
    pair_a = joint_pairs[i]
    pair_b = joint_pairs[i+1]
    try:
        distance1 = math.dist(points[pair_a[0]], points[pair_a[1]]) / math.dist(points[pair_b[0]], points[pair_b[1]])
        distance2 = math.dist(scaled_points[pair_a[0]], scaled_points[pair_a[1]]) / math.dist(scaled_points[pair_b[0]], scaled_points[pair_b[1]])

        deviations.append(abs(distance1 - distance2))
    except:
        pass

print(f"Average deviation of the ratio between distances of key joint pairs after scaling: {round(statistics.mean(deviations), 4)}")

print("---- HASHING ----")

# Resize the scaled image to the original size
scaled = cv2.resize(org, (0,0), fx=2, fy=2) 

print(f"The sha256 value of the original image is: \n{hashlib.sha256((bytes(org))).hexdigest()}\n")
print(f"The sha256 value of the scaled image after re-scaling to the original size is: \n{hashlib.sha256((bytes(scaled))).hexdigest()}\n")

cv2.imshow("Output-Keypoints", frame) 
cv2.waitKey(0) 
cv2.destroyAllWindows()

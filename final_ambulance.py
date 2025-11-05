# LAG
# NO. OF VEHICLES IN SIGNAL CLASS
# stops not used
# DISTRIBUTION
# BUS TOUCHING ON TURNS
# Distribution using python class

# *** IMAGE XY COOD IS TOP LEFT
import random
import math
import time
import threading
import pygame
import sys
import os
import numpy as np
from traffic_management_system import TrafficManagementSystem

# Try to import OpenCV and the ambulance detector
try:
    import cv2
    from ambulance_contour_detector import AmbulanceContourDetector
    cv2_available = True
except ImportError:
    print("Warning: OpenCV or ambulance detector not available. Running without ambulance detection.")
    cv2_available = False

# Initialize ambulance detector if model is available
ambulance_detector = None
model_path = "ambulance_detector_resnet50.pth"
if cv2_available:
    try:
        if os.path.exists(model_path):
            ambulance_detector = AmbulanceContourDetector(model_path)
            print(f"Ambulance detector loaded successfully from {model_path}")
        else:
            print(f"Warning: Ambulance detector model not found at {model_path}")
    except Exception as e:
        print(f"Error loading ambulance detector: {e}")
        # Continue without detector

# Default values of signal times
defaultRed = 150
defaultYellow = 7
defaultGreen = 15
defaultMinimum = 10
defaultMaximum = 60
tms = TrafficManagementSystem()

# Ambulance related variables
ambulanceGenerationTime = 50  # Generate ambulance every 50 seconds
firstAmbulanceTime = 30  # First ambulance appears at 30 seconds
ambulanceYellowTime = 3  # Buffer time for ambulance priority
ambulancesDetected = []  # List to store detected ambulances: [{'lane': lane_number, 'time': detection_time}, ...]
ambulanceLogs = []  # To store logs of ambulance events
isAmbulancePriority = False  # Flag to indicate if ambulance priority is active
interruptedLane = None  # Store the lane that was interrupted
remainingGreenTime = 0  # Remaining green time for interrupted lane

signals = []
noOfSignals = 4
simTime = 300       # change this to change time of simulation
timeElapsed = 0

currentGreen = 0   # Indicates which signal is green
nextGreen = (currentGreen+1)%noOfSignals
currentYellow = 0   # Indicates whether yellow signal is on or off 

# Average times for vehicles to pass the intersection
carTime = 2
bikeTime = 1
rickshawTime = 2.25 
busTime = 2.5
truckTime = 2.5

detection_info = {
    'lane': None,
    'timestamp': None,
    'vehicle_counts': {
        'cars': 0,
        'bikes': 0,
        'buses': 0,
        'trucks': 0,
        'rickshaws': 0
    },
    'allocated_time': 0,
    'last_detection_time': 0
}

# Count of cars at a traffic signal
noOfCars = 0
noOfBikes = 0
noOfBuses =0
noOfTrucks = 0
noOfRickshaws = 0
noOfLanes = 2

# Red signal time at which cars will be detected at a signal
detectionTime = 10

speeds = {'car':0.6, 'bus':0.5, 'truck':0.5, 'rickshaw':0.6, 'bike':0.7, 'ambulance':1.0}  # Ambulance is faster

# Coordinates of start
x = {'right':[0,0,0], 'down':[755,727,697], 'left':[1400,1400,1400], 'up':[602,627,657]}    
y = {'right':[348,370,398], 'down':[0,0,0], 'left':[498,466,436], 'up':[800,800,800]}

# vehicles = {'right': {0:[], 1:[], 2:[], 'crossed':0}, 'down': {0:[], 1:[], 2:[], 'crossed':0}, 'left': {0:[], 1:[], 2:[], 'crossed':0}, 'up': {0:[], 1:[], 2:[], 'crossed':0}}
vehicles = {'right': {0:[], 1:[], 2:[]}, 'down': {0:[], 1:[], 2:[]}, 'left': {0:[], 1:[], 2:[]}, 'up': {0:[], 1:[], 2:[]}} 
vehicleTypes = {0:'car', 1:'bus', 2:'truck', 3:'rickshaw', 4:'bike', 5:'ambulance'}  # Added ambulance as type 5
directionNumbers = {0:'right', 1:'down', 2:'left', 3:'up'}

# Coordinates of signal image, timer, and vehicle count
signalCoods = [(530,230),(810,230),(810,570),(530,570)]
signalTimerCoods = [(530,210),(810,210),(810,550),(530,550)]
vehicleCountCoods = [(480,210),(880,210),(880,550),(480,550)]
vehicleCountTexts = ["0", "0", "0", "0"]
# New coordinates for the waiting vehicles label
waitingLabelCoods = [(380,180),(980,180),(980,580),(380,580)]

# New coordinates for the waiting vehicles count
waitingCountCoods = [(480,180),(880,180),(880,580),(480,580)]

# Coordinates of stop lines
stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}
stops = {'right': [580,580,580], 'down': [320,320,320], 'left': [810,810,810], 'up': [545,545,545]}

mid = {'right': {'x':705, 'y':445}, 'down': {'x':695, 'y':450}, 'left': {'x':695, 'y':425}, 'up': {'x':695, 'y':400}}
rotationAngle = 3

# Gap between vehicles
gap = 15    # stopping gap
gap2 = 15   # moving gap

pygame.init()
simulation = pygame.sprite.Group()

class TrafficSignal:
    def __init__(self, red, yellow, green, minimum, maximum):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.minimum = minimum
        self.maximum = maximum
        self.signalText = "30"
        self.totalGreenTime = 0
        
class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction, will_turn):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        self.willTurn = will_turn
        self.turned = 0
        self.rotateAngle = 0
        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1
        
        # Path to vehicle image - handling ambulance images
        if vehicleClass == 'ambulance':
            # Ambulance images should be in the same directory structure as other vehicles
            path = "images/" + direction + "/" + vehicleClass + ".png"
            # Check if ambulance image exists
            if not os.path.exists(path):
                # Fallback to car image if ambulance image doesn't exist
                path = "images/" + direction + "/car.png"
                print(f"Warning: Ambulance image not found at {path}, using car image instead")
        else:
            path = "images/" + direction + "/" + vehicleClass + ".png"
            
        self.originalImage = pygame.image.load(path)
        self.currentImage = pygame.image.load(path)

    
        if(direction=='right'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):    # if more than 1 vehicle in the lane of vehicle before it has crossed stop line
                self.stop = vehicles[direction][lane][self.index-1].stop - vehicles[direction][lane][self.index-1].currentImage.get_rect().width - gap         # setting stop coordinate as: stop coordinate of next vehicle - width of next vehicle - gap
            else:
                self.stop = defaultStop[direction]
            # Set new starting and stopping coordinate
            temp = self.currentImage.get_rect().width + gap    
            x[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif(direction=='left'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):
                self.stop = vehicles[direction][lane][self.index-1].stop + vehicles[direction][lane][self.index-1].currentImage.get_rect().width + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap
            x[direction][lane] += temp
            stops[direction][lane] += temp
        elif(direction=='down'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):
                self.stop = vehicles[direction][lane][self.index-1].stop - vehicles[direction][lane][self.index-1].currentImage.get_rect().height - gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif(direction=='up'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):
                self.stop = vehicles[direction][lane][self.index-1].stop + vehicles[direction][lane][self.index-1].currentImage.get_rect().height + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] += temp
            stops[direction][lane] += temp
        simulation.add(self)

    def render(self, screen):
        screen.blit(self.currentImage, (self.x, self.y))

    def move(self):
        if(self.direction=='right'):
            if(self.crossed==0 and self.x+self.currentImage.get_rect().width>stopLines[self.direction]):   # if the image has crossed stop line now
                self.crossed = 1
                # vehicles[self.direction]['crossed'] += 1
            if(self.willTurn==1):
                if(self.crossed==0 or self.x+self.currentImage.get_rect().width<mid[self.direction]['x']):
                    if((self.x+self.currentImage.get_rect().width<=self.stop or (currentGreen==0 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.x+self.currentImage.get_rect().width<(vehicles[self.direction][self.lane][self.index-1].x - gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):                
                        self.x += self.speed
                else:   
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 2
                        self.y += 1.8
                        if(self.rotateAngle==90):
                            self.turned = 1
                            # path = "images/" + directionNumbers[((self.direction_number+1)%noOfSignals)] + "/" + self.vehicleClass + ".png"
                            # self.x = mid[self.direction]['x']
                            # self.y = mid[self.direction]['y']
                            # self.image = pygame.image.load(path)
                    else:
                        if(self.index==0 or self.y+self.currentImage.get_rect().height<(vehicles[self.direction][self.lane][self.index-1].y - gap2) or self.x+self.currentImage.get_rect().width<(vehicles[self.direction][self.lane][self.index-1].x - gap2)):
                            self.y += self.speed
            else: 
                if((self.x+self.currentImage.get_rect().width<=self.stop or self.crossed == 1 or (currentGreen==0 and currentYellow==0)) and (self.index==0 or self.x+self.currentImage.get_rect().width<(vehicles[self.direction][self.lane][self.index-1].x - gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                # (if the image has not reached its stop coordinate or has crossed stop line or has green signal) and (it is either the first vehicle in that lane or it is has enough gap to the next vehicle in that lane)
                    self.x += self.speed  # move the vehicle



        elif(self.direction=='down'):
            if(self.crossed==0 and self.y+self.currentImage.get_rect().height>stopLines[self.direction]):
                self.crossed = 1
                # vehicles[self.direction]['crossed'] += 1
            if(self.willTurn==1):
                if(self.crossed==0 or self.y+self.currentImage.get_rect().height<mid[self.direction]['y']):
                    if((self.y+self.currentImage.get_rect().height<=self.stop or (currentGreen==1 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.y+self.currentImage.get_rect().height<(vehicles[self.direction][self.lane][self.index-1].y - gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):                
                        self.y += self.speed
                else:   
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 2.5
                        self.y += 2
                        if(self.rotateAngle==90):
                            self.turned = 1
                    else:
                        if(self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or self.y<(vehicles[self.direction][self.lane][self.index-1].y - gap2)):
                            self.x -= self.speed
            else: 
                if((self.y+self.currentImage.get_rect().height<=self.stop or self.crossed == 1 or (currentGreen==1 and currentYellow==0)) and (self.index==0 or self.y+self.currentImage.get_rect().height<(vehicles[self.direction][self.lane][self.index-1].y - gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                    self.y += self.speed
            
        elif(self.direction=='left'):
            if(self.crossed==0 and self.x<stopLines[self.direction]):
                self.crossed = 1
                # vehicles[self.direction]['crossed'] += 1
            if(self.willTurn==1):
                if(self.crossed==0 or self.x>mid[self.direction]['x']):
                    if((self.x>=self.stop or (currentGreen==2 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):                
                        self.x -= self.speed
                else: 
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 1.8
                        self.y -= 2.5
                        if(self.rotateAngle==90):
                            self.turned = 1
                            # path = "images/" + directionNumbers[((self.direction_number+1)%noOfSignals)] + "/" + self.vehicleClass + ".png"
                            # self.x = mid[self.direction]['x']
                            # self.y = mid[self.direction]['y']
                            # self.currentImage = pygame.image.load(path)
                    else:
                        if(self.index==0 or self.y>(vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height +  gap2) or self.x>(vehicles[self.direction][self.lane][self.index-1].x + gap2)):
                            self.y -= self.speed
            else: 
                if((self.x>=self.stop or self.crossed == 1 or (currentGreen==2 and currentYellow==0)) and (self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                # (if the image has not reached its stop coordinate or has crossed stop line or has green signal) and (it is either the first vehicle in that lane or it is has enough gap to the next vehicle in that lane)
                    self.x -= self.speed  # move the vehicle    
            # if((self.x>=self.stop or self.crossed == 1 or (currentGreen==2 and currentYellow==0)) and (self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2))):                
            #     self.x -= self.speed
        elif(self.direction=='up'):
            if(self.crossed==0 and self.y<stopLines[self.direction]):
                self.crossed = 1
                # vehicles[self.direction]['crossed'] += 1
            if(self.willTurn==1):
                if(self.crossed==0 or self.y>mid[self.direction]['y']):
                    if((self.y>=self.stop or (currentGreen==3 and currentYellow==0) or self.crossed == 1) and (self.index==0 or self.y>(vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height +  gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):
                        self.y -= self.speed
                else:   
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 1
                        self.y -= 1
                        if(self.rotateAngle==90):
                            self.turned = 1
                    else:
                        if(self.index==0 or self.x<(vehicles[self.direction][self.lane][self.index-1].x - vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width - gap2) or self.y>(vehicles[self.direction][self.lane][self.index-1].y + gap2)):
                            self.x += self.speed
            else: 
                if((self.y>=self.stop or self.crossed == 1 or (currentGreen==3 and currentYellow==0)) and (self.index==0 or self.y>(vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height + gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                    self.y -= self.speed

# Check for ambulances in all lanes and handle priority
def checkForAmbulances():
    global ambulancesDetected, currentGreen, isAmbulancePriority, interruptedLane, remainingGreenTime, currentYellow, nextGreen
    
    # Check all lanes for ambulances
    for lane_num in range(noOfSignals):
        direction = directionNumbers[lane_num]
        
        # Check all vehicles in all lanes (ambulances can be in any lane)
        for lane in range(3):
            for vehicle in vehicles[direction][lane]:
                if vehicle.vehicleClass == 'ambulance' and vehicle.crossed == 0:
                    # If we detect an ambulance, add it to the detected list if not already there
                    ambulance_already_detected = False
                    for amb in ambulancesDetected:
                        if amb['lane'] == lane_num and amb['id'] == id(vehicle):
                            ambulance_already_detected = True
                            break
                    
                    if not ambulance_already_detected:
                        # Add new ambulance to detected list
                        ambulancesDetected.append({
                            'lane': lane_num,
                            'time': timeElapsed,
                            'id': id(vehicle),
                            'processed': False
                        })
                        log_message = f"Ambulance detected in Lane {lane_num+1} (direction: {direction}) at time {timeElapsed}s"
                        print(log_message)
                        ambulanceLogs.append(log_message)

    # Process ambulances by time detected (first detected, first served)
    if ambulancesDetected and not isAmbulancePriority:
        # Sort ambulances by detection time
        ambulancesDetected.sort(key=lambda x: x['time'])
        
        # Process the earliest ambulance that hasn't been processed yet
        for ambulance in ambulancesDetected:
            if not ambulance['processed']:
                lane_with_ambulance = ambulance['lane']
                
                # Debug logging for ambulance priority
                debug_message = f"DEBUG: Processing ambulance in lane {lane_with_ambulance+1}, current green: {currentGreen+1}"
                print(debug_message)
                
                # If ambulance is already in green lane, do nothing special
                if lane_with_ambulance == currentGreen:
                    log_message = f"Ambulance in Lane {lane_with_ambulance+1} already has green signal at time {timeElapsed}s"
                    print(log_message)
                    ambulanceLogs.append(log_message)
                    ambulance['processed'] = True
                else:
                    # Need to interrupt current signal and give priority to ambulance lane
                    isAmbulancePriority = True
                    interruptedLane = currentGreen
                    remainingGreenTime = max(signals[currentGreen].green, 0)  # Fix negative time issue
                    
                    # Log the interruption
                    log_message = f"Interrupting Lane {currentGreen+1} to give priority to ambulance in Lane {lane_with_ambulance+1} at time {timeElapsed}s. Remaining green time: {remainingGreenTime}s"
                    print(log_message)
                    ambulanceLogs.append(log_message)
                    
                    # Force immediate transition to ambulance lane
                    # First, set all signals to red except the ambulance lane
                    for i in range(noOfSignals):
                        if i != lane_with_ambulance:
                            signals[i].red = 1  # Will be updated later

                    # Force yellow for current green immediately
                    signals[currentGreen].green = 0  # End green immediately
                    signals[currentGreen].yellow = ambulanceYellowTime

                    # Mark that we're in yellow phase
                    currentYellow = 1
                    
                    # Set next green to ambulance lane
                    nextGreen = lane_with_ambulance
                    
                    # Calculate green time for ambulance lane
                    green_time = calculateEmergencyGreenTime(lane_with_ambulance)
                    
                    # Update red times for all other signals
                    for i in range(noOfSignals):
                        if i != currentGreen and i != lane_with_ambulance:
                            signals[i].red = ambulanceYellowTime + green_time
                    
                    # Prepare ambulance lane to be ready after yellow
                    signals[lane_with_ambulance].red = ambulanceYellowTime
                    signals[lane_with_ambulance].green = green_time
                    
                    debug_message = f"DEBUG CRITICAL: Ambulance lane {lane_with_ambulance+1} will get green after {ambulanceYellowTime}s yellow, green time: {green_time}s"
                    print(debug_message)
                    
                    ambulance['processed'] = True
                    return  # Exit after handling one ambulanceExit after handling one ambulance

# Handle emergency transition to ambulance lane
def emergencyTransition(ambulance_lane):
    global currentGreen, nextGreen, currentYellow, isAmbulancePriority
    
    # Log this emergency transition
    log_message = f"EMERGENCY TRANSITION: Giving priority to lane {ambulance_lane+1} at time {timeElapsed}s"
    print(log_message)
    ambulanceLogs.append(log_message)
    
    # Force yellow signal for current green lane
    signals[currentGreen].yellow = ambulanceYellowTime
    currentYellow = 1
    
    # Explicitly set the ambulance lane as the next green signal
    nextGreen = ambulance_lane
    
    # Calculate emergency green time for ambulance lane
    calculateEmergencyGreenTime(ambulance_lane)
    
    # Set the priority flag to true
    isAmbulancePriority = True

# Calculate green time for ambulance lane
def calculateEmergencyGreenTime(ambulance_lane):
    # Get vehicle counts for ambulance lane
    vehicle_counts = {
        'cars': 0,
        'bikes': 0,
        'buses': 0,
        'trucks': 0,
        'rickshaws': 0
    }
    
    # Count vehicles in each lane for the ambulance direction
    direction = directionNumbers[ambulance_lane]
    
    # Count bikes in lane 0
    for vehicle in vehicles[direction][0]:
        if vehicle.crossed == 0:
            vehicle_counts['bikes'] += 1
    
    # Count other vehicles in lanes 1-2
    for lane in range(1, 3):
        for vehicle in vehicles[direction][lane]:
            if vehicle.crossed == 0:
                if vehicle.vehicleClass == 'car':
                    vehicle_counts['cars'] += 1
                elif vehicle.vehicleClass == 'bus':
                    vehicle_counts['buses'] += 1
                elif vehicle.vehicleClass == 'truck':
                    vehicle_counts['trucks'] += 1
                elif vehicle.vehicleClass == 'rickshaw':
                    vehicle_counts['rickshaws'] += 1
    
    # Get current hour and day
    current_hour = int(time.strftime('%H'))
    day_of_week = int(time.strftime('%w'))
    
    # Process lane with traffic management system
    lane_state = tms.process_lane(
        ambulance_lane,
        simulate=False,
        vehicle_counts=vehicle_counts,
        hour=current_hour,
        day_of_week=day_of_week
    )
    
    # Get calculated green time and ensure it's at least the minimum required
    green_time = lane_state['allocated_time']
    green_time = max(defaultMinimum, min(defaultMaximum, green_time))
    
    # Set this as the green time for the ambulance lane
    signals[ambulance_lane].green = green_time
    
    # Update detection info
    detection_info.update({
        'lane': ambulance_lane,
        'timestamp': time.time(),
        'vehicle_counts': vehicle_counts,
        'allocated_time': green_time,
        'last_detection_time': time.time()
    })
    
    log_message = f"Emergency green time for Lane {ambulance_lane+1}: {green_time}s with vehicle count: {vehicle_counts}"
    print(log_message)
    ambulanceLogs.append(log_message)
    
    # Return the calculated green time
    return green_time

# Calculate green time for a specific lane (used when resuming interrupted lanes)
def calculateGreenTimeForLane(lane_num):
    # Get vehicle counts for the specified lane
    vehicle_counts = {
        'cars': 0,
        'bikes': 0,
        'buses': 0,
        'trucks': 0,
        'rickshaws': 0
    }
    
    direction = directionNumbers[lane_num]
    
    # Count bikes in lane 0
    for vehicle in vehicles[direction][0]:
        if vehicle.crossed == 0:
            vehicle_counts['bikes'] += 1
    
    # Count other vehicles in lanes 1-2
    for lane in range(1, 3):
        for vehicle in vehicles[direction][lane]:
            if vehicle.crossed == 0:
                vclass = vehicle.vehicleClass
                if vclass == 'car':
                    vehicle_counts['cars'] += 1
                elif vclass == 'bus':
                    vehicle_counts['buses'] += 1
                elif vclass == 'truck':
                    vehicle_counts['trucks'] += 1
                elif vclass == 'rickshaw':
                    vehicle_counts['rickshaws'] += 1
    
    # Get current hour and day
    current_hour = int(time.strftime('%H'))
    day_of_week = int(time.strftime('%w'))
    
    # Process lane with traffic management system
    lane_state = tms.process_lane(
        lane_num,
        simulate=False,
        vehicle_counts=vehicle_counts,
        hour=current_hour,
        day_of_week=day_of_week
    )
    
    # Get calculated green time and ensure it's within bounds
    green_time = lane_state['allocated_time']
    green_time = max(defaultMinimum, min(defaultMaximum, green_time))
    
    # Set the green time for the lane
    signals[lane_num].green = green_time
    
    log_message = f"Recalculated green time for Lane {lane_num+1}: {green_time}s with vehicle count: {vehicle_counts}"
    print(log_message)
    ambulanceLogs.append(log_message)
    
    return green_time

def isAmbulanceInLane(lane_num):
    direction = directionNumbers[lane_num]
    for lane in range(3):
        for vehicle in vehicles[direction][lane]:
            if vehicle.vehicleClass == 'ambulance' and vehicle.crossed == 0:
                return True
    return False

# Helper function to check if all ambulances have passed
def allAmbulancesPassed():
    for ambulance in ambulancesDetected:
        lane_num = ambulance['lane']
        if isAmbulanceInLane(lane_num):
            return False
    return True

# Function to detect ambulances using camera images
def detect_ambulances_from_cameras():
    global ambulancesDetected
    
    # Skip if ambulance detector isn't available
    if ambulance_detector is None:
        return
    
    # Placeholder for camera inputs
    # In a real system, we would get images from actual cameras
    # For simulation purposes, we'll assume cameras are monitoring each lane
    
    # For each lane (direction), check for ambulances
    for lane_num in range(noOfSignals):
        direction = directionNumbers[lane_num]
        
        try:
            # Placeholder for getting camera image
            # In a real system, this would be: image = get_camera_feed(lane_num)
            # For simulation, we'll create a blank image and overlay any ambulances in the lane
            image = create_lane_image(lane_num)
            
            if image is not None:
                # Detect ambulance in the image using our detector
                is_ambulance, confidence = ambulance_detector.detect_ambulance(image)
                
                if is_ambulance:
                    # If ambulance detected in camera, add it to detected list with high confidence
                    ambulance_already_detected = False
                    
                    # Check if this ambulance has already been detected
                    for amb in ambulancesDetected:
                        if amb['lane'] == lane_num and timeElapsed - amb['time'] < 10:
                            ambulance_already_detected = True
                            break
                    
                    if not ambulance_already_detected:
                        # Add new ambulance to detected list
                        ambulancesDetected.append({
                            'lane': lane_num,
                            'time': timeElapsed,
                            'id': f"camera_{lane_num}_{timeElapsed}",
                            'processed': False,
                            'confidence': confidence
                        })
                        
                        log_message = f"Ambulance detected by camera in Lane {lane_num+1} (direction: {direction}) at time {timeElapsed}s with confidence {confidence:.2f}"
                        print(log_message)
                        ambulanceLogs.append(log_message)
        
        except Exception as e:
            print(f"Error processing camera feed for lane {lane_num}: {e}")

# Create a simulated lane image for testing the detector
def create_lane_image(lane_num):
    # This is a placeholder function that creates a blank image
    # In a real system, this would be replaced with actual camera feed
    
    # Create a blank image (black background, 640x480)
    h, w = 480, 640
    image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Get the direction for this lane
    direction = directionNumbers[lane_num]
    
    # Check if there are ambulances in this lane
    has_ambulance = False
    for vehicle in vehicles[direction][1]:  # Check middle lane where ambulances travel
        if vehicle.vehicleClass == 'ambulance' and vehicle.crossed == 0:
            has_ambulance = True
            break
    
    if has_ambulance:
        # If there's an ambulance, add a simple ambulance representation to the image
        # Draw a white rectangle with red siren lights
        # This is just to simulate what a camera might see
        ambulance_x = w // 2 - 50
        ambulance_y = h // 2 - 30
        ambulance_w = 100
        ambulance_h = 60
        
        # White body
        cv2.rectangle(image, (ambulance_x, ambulance_y), 
                     (ambulance_x + ambulance_w, ambulance_y + ambulance_h), 
                     (255, 255, 255), -1)
        
        # Red siren light
        cv2.circle(image, (ambulance_x + ambulance_w // 2, ambulance_y), 10, (0, 0, 255), -1)
        
        # Add some text
        cv2.putText(image, "AMBULANCE", 
                   (ambulance_x + 10, ambulance_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return image

# Initialization of signals with default values
def initialize():
    ts1 = TrafficSignal(0, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts1)
    ts2 = TrafficSignal(ts1.red+ts1.yellow+ts1.green, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts4)
    repeat()

# Set time according to formula
def setTime():
    global detection_info
    
    # Get vehicle counts for nextGreen lane from the simulation
    vehicle_counts = {
        'cars': 0,
        'bikes': 0,
        'buses': 0,
        'trucks': 0,
        'rickshaws': 0
    }
    
    # Count bikes in lane 0
    for j in range(len(vehicles[directionNumbers[nextGreen]][0])):
        vehicle = vehicles[directionNumbers[nextGreen]][0][j]
        if(vehicle.crossed==0):
            vehicle_counts['bikes'] += 1
    
    # Count other vehicles in lanes 1-2
    for i in range(1,3):
        for j in range(len(vehicles[directionNumbers[nextGreen]][i])):
            vehicle = vehicles[directionNumbers[nextGreen]][i][j]
            if(vehicle.crossed==0):
                vclass = vehicle.vehicleClass
                if(vclass=='car'):
                    vehicle_counts['cars'] += 1
                elif(vclass=='bus'):
                    vehicle_counts['buses'] += 1
                elif(vclass=='truck'):
                    vehicle_counts['trucks'] += 1
                elif(vclass=='rickshaw'):
                    vehicle_counts['rickshaws'] += 1
    
    # Get current hour and day
    current_hour = int(time.strftime('%H'))
    day_of_week = int(time.strftime('%w'))  # 0=Sunday, 6=Saturday
    
    # Process lane with the traffic management system
    lane_state = tms.process_lane(
        nextGreen, 
        simulate=False,  # Use actual counts from simulation
        vehicle_counts=vehicle_counts,
        hour=current_hour,
        day_of_week=day_of_week
    )
    
    # Set the green time for the next signal
    greenTime = lane_state['allocated_time']
    
    # Update detection info with more explicit information
    detection_info.update({
        'lane': nextGreen,
        'timestamp': time.time(),
        'vehicle_counts': vehicle_counts,
        'allocated_time': greenTime,
        'last_detection_time': time.time()
    })
    
    print(f"Detection Info: Lane {nextGreen}, Vehicles: {vehicle_counts}, Allocated Time: {greenTime}")
    
    # Rest of the existing setTime function remains the same
    if(greenTime < defaultMinimum):
        greenTime = defaultMinimum
    elif(greenTime > defaultMaximum):
        greenTime = defaultMaximum
        
    signals[(currentGreen+1)%(noOfSignals)].green = greenTime

   
def repeat():
    global currentGreen, currentYellow, nextGreen, isAmbulancePriority, interruptedLane, remainingGreenTime
    
    # Before normal traffic flow, check for ambulances
    checkForAmbulances()
    
    # Also check for ambulances using camera detection
    if cv2_available:
        detect_ambulances_from_cameras()
    
    while(signals[currentGreen].green > 0):   # while the timer of current green signal is not zero
        # Check for ambulances on each tick
        checkForAmbulances()
        
        # Check for ambulances using camera every few seconds
        if cv2_available and timeElapsed % 5 == 0:  # Check every 5 seconds to avoid too much processing
            detect_ambulances_from_cameras()
        
        printStatus()
        updateValues()
        
        # If this is the time to set detection for next signal
        if(signals[(currentGreen+1)%(noOfSignals)].red == detectionTime):
            thread = threading.Thread(name="detection", target=setTime, args=())
            thread.daemon = True
            thread.start()
            
        time.sleep(1)
    
    currentYellow = 1   # set yellow signal on
    vehicleCountTexts[currentGreen] = "0"
    
    # reset stop coordinates of lanes and vehicles 
    for i in range(0,3):
        stops[directionNumbers[currentGreen]][i] = defaultStop[directionNumbers[currentGreen]]
        for vehicle in vehicles[directionNumbers[currentGreen]][i]:
            vehicle.stop = defaultStop[directionNumbers[currentGreen]]
    
    # Check for ambulances during yellow phase
    while(signals[currentGreen].yellow > 0):  # while the timer of current yellow signal is not zero
        checkForAmbulances()
        printStatus()
        updateValues()
        time.sleep(1)
        
    currentYellow = 0   # set yellow signal off
    
    # Store previous green for reference
    previousGreen = currentGreen
    
    # Check if we're handling ambulance priority and need to restore interrupted lane
    if isAmbulancePriority and interruptedLane is not None and currentGreen != interruptedLane:
        # Check if we just finished the ambulance's lane
        ambulance_present = False
        for lane in range(3):
            for vehicle in vehicles[directionNumbers[currentGreen]][lane]:
                if vehicle.vehicleClass == 'ambulance' and vehicle.crossed == 0:
                    ambulance_present = True
                    break
            if ambulance_present:
                break
        
        # If no more ambulances in the current lane, restore to interrupted lane
        if not ambulance_present:
            log_message = f"Ambulance priority complete for Lane {currentGreen+1}, returning to interrupted Lane {interruptedLane+1} at time {timeElapsed}s"
            print(log_message)
            ambulanceLogs.append(log_message)
            
            # Set the next green to the interrupted lane
            nextGreen = interruptedLane
            
            # Calculate new green time for the interrupted lane
            green_time = calculateGreenTimeForLane(interruptedLane)
            if remainingGreenTime > 0:
                # Use the remaining time if it was interrupted, but ensure it's at least the minimum
                green_time = max(remainingGreenTime, defaultMinimum)
                log_message = f"Restored remaining green time: {green_time}s for Lane {interruptedLane+1}"
                print(log_message)
                ambulanceLogs.append(log_message)
            
            # Update the signal timing
            signals[interruptedLane].green = green_time
            
            # Reset ambulance priority state
            isAmbulancePriority = False
            interruptedLane = None
            remainingGreenTime = 0
        
    # If no special handling for ambulance, proceed normally
    if nextGreen == (previousGreen+1) % noOfSignals:  # Normal cycle
        # Reset all signal times of current signal to default times
        signals[currentGreen].green = defaultGreen
        signals[currentGreen].yellow = defaultYellow
        signals[currentGreen].red = defaultRed
    
    currentGreen = nextGreen  # set next signal as green signal
    
    # Only change nextGreen if we're not in ambulance priority mode
    if not isAmbulancePriority:
        nextGreen = (currentGreen+1) % noOfSignals    # set next green signal
    
    # Update red times for the next signals
    signals[nextGreen].red = signals[currentGreen].yellow + signals[currentGreen].green
    
    # Continue simulation
    repeat()

# Print the signal timers on cmd
def printStatus():                                                                                           
    for i in range(0, noOfSignals):
        if(i==currentGreen):
            if(currentYellow==0):
                print("GREEN TS",i+1,"-> r:",signals[i].red," y:",signals[i].yellow," g:",signals[i].green)
            else:
                print("YELLOW TS",i+1,"-> r:",signals[i].red," y:",signals[i].yellow," g:",signals[i].green)
        else:
            print("RED TS",i+1,"-> r:",signals[i].red," y:",signals[i].yellow," g:",signals[i].green)
        
        # Calculate current waiting vehicles
        current_count = 0
        for lane in range(3):
            for vehicle in vehicles[directionNumbers[i]][lane]:
                if vehicle.crossed == 0:
                    current_count += 1
        print(f"TS{i+1} Current waiting vehicles: {current_count}")
    print()

def getCurrentVehicleCount(direction):
    count = 0
    for lane in range(3):
        for vehicle in vehicles[direction][lane]:
            if vehicle.crossed == 0:
                count += 1
    return count

# Update values of the signal timers after every second
def updateValues():
    for i in range(0, noOfSignals):
        if(i == currentGreen):
            if(currentYellow == 0):
                if signals[i].green > 0:  # Prevent negative values
                    signals[i].green -= 1
                    signals[i].totalGreenTime += 1
            else:
                if signals[i].yellow > 0:  # Prevent negative values
                    signals[i].yellow -= 1
        else:
            if signals[i].red > 0:  # Prevent negative values
                signals[i].red -= 1

# Generating vehicles in the simulation
def generateVehicles():
    while(True):
        vehicle_type = random.randint(0,4)
        if(vehicle_type==4):
            lane_number = 0
        else:
            lane_number = random.randint(0,1) + 1
        will_turn = 0
        if(lane_number==2):
            temp = random.randint(0,4)
            if(temp<=2):
                will_turn = 1
            elif(temp>2):
                will_turn = 0
        temp = random.randint(0,999)
        direction_number = 0
        a = [400,800,900,1000]
        if(temp<a[0]):
            direction_number = 0
        elif(temp<a[1]):
            direction_number = 1
        elif(temp<a[2]):
            direction_number = 2
        elif(temp<a[3]):
            direction_number = 3
        Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number, directionNumbers[direction_number], will_turn)
        time.sleep(0.75)

# Generating ambulances at specific intervals
def generateAmbulances():
    global timeElapsed, firstAmbulanceTime, ambulanceGenerationTime
    
    # Wait for the first ambulance time
    print(f"Waiting for first ambulance at {firstAmbulanceTime}s")
    while timeElapsed < firstAmbulanceTime:
        time.sleep(1)
    
    # Generate first ambulance
    generateAmbulance()
    last_generation_time = timeElapsed
    print(f"First ambulance generated at {timeElapsed}s")
    
    # Continue generating ambulances at regular intervals
    while True:
        time.sleep(1)
        if timeElapsed - last_generation_time >= ambulanceGenerationTime:
            generateAmbulance()
            last_generation_time = timeElapsed
            print(f"Ambulance generated at interval: {timeElapsed}s")

def generateAmbulance():
    # Select a random direction for the ambulance
    direction_number = random.randint(0, 3)
    direction = directionNumbers[direction_number]
    
    # Ambulances use lane 1 (middle lane)
    lane_number = 1
    
    # Ambulances don't turn in this simulation
    will_turn = 0
    
    try:
        # Create the ambulance vehicle
        vehicleClass = 'ambulance'
        ambulance = Vehicle(lane_number, vehicleClass, direction_number, direction, will_turn)
        
        # Log the ambulance generation
        log_message = f"Ambulance generated in lane {direction_number+1} (direction: {direction}) at time {timeElapsed}s"
        print(log_message)
        ambulanceLogs.append(log_message)
        
        return ambulance
    except Exception as e:
        print(f"Error generating ambulance: {e}")
        return None

def simulationTime():
    global timeElapsed, simTime
    while(True):
        timeElapsed += 1
        time.sleep(1)
        if(timeElapsed==simTime):
            totalVehicles = 0
            print('Lane-wise Vehicle Counts:')
            
            # Count crossed vehicles in each direction
            for i in range(noOfSignals):
                direction = directionNumbers[i]
                crossed_count = 0
                for lane in range(3):
                    for vehicle in vehicles[direction][lane]:
                        if vehicle.crossed == 1:
                            crossed_count += 1
                
                print(f'Lane {i+1} ({direction}): {crossed_count}')
                totalVehicles += crossed_count
                
            # Print ambulance statistics if any were detected
            if ambulanceLogs:
                print('\nAmbulance Priority Events:')
                for log in ambulanceLogs:
                    print(log)
                    
            print('\nTotal vehicles passed:', totalVehicles)
            print('Total time passed:', timeElapsed)
            print('No. of vehicles passed per unit time:', (float(totalVehicles)/float(timeElapsed)))
            os._exit(1)

class Main:
    global tms
    tms.setup(train_new_model=False)
    thread4 = threading.Thread(name="simulationTime",target=simulationTime, args=()) 
    thread4.daemon = True
    thread4.start()

    thread2 = threading.Thread(name="initialization",target=initialize, args=())    # initialization
    thread2.daemon = True
    thread2.start()

    # Start the ambulance generation thread
    thread5 = threading.Thread(name="generateAmbulances", target=generateAmbulances, args=())
    thread5.daemon = True
    thread5.start()
    print("Starting ambulance generation thread...")

    # Colours 
    black = (0, 0, 0)
    white = (255, 255, 255)
    gray = (200, 200, 200)

    # Screensize 
    screenWidth = 1400
    screenHeight = 800
    screenSize = (screenWidth, screenHeight)

    # Setting background image i.e. image of intersection
    background = pygame.image.load('images/mod_int_snow.png')

    screen = pygame.display.set_mode(screenSize)
    pygame.display.set_caption("SIMULATION")

    # Loading signal images and font
    redSignal = pygame.image.load('images/signals/red.png')
    yellowSignal = pygame.image.load('images/signals/yellow.png')
    greenSignal = pygame.image.load('images/signals/green.png')
    font = pygame.font.Font(None, 30)
    small_font = pygame.font.Font(None, 24)
    countLabel = [
    font.render("Waiting Vehicles:", True, white, black),
    font.render("Waiting Vehicles:", True, white, black),
    font.render("Waiting Vehicles:", True, white, black),
    font.render("Waiting Vehicles:", True, white, black)
]

    thread3 = threading.Thread(name="generateVehicles",target=generateVehicles, args=())    # Generating vehicles
    thread3.daemon = True
    thread3.start()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        screen.blit(background,(0,0))   # display background in simulation
        
        # Display ambulance status
        ambulance_status_box = pygame.Surface((300, 80), pygame.SRCALPHA)
        ambulance_status_box.fill((0, 0, 0, 200))  # Semi-transparent black background
        screen.blit(ambulance_status_box, (screenWidth - 320, 10))
        
        # Display ambulance information
        amb_title = font.render("Ambulance Status", True, (255, 255, 0))
        screen.blit(amb_title, (screenWidth - 300, 20))
        
        # Count active ambulances in the simulation
        active_ambulances = 0
        for direction in directionNumbers.values():
            for lane in range(3):
                for vehicle in vehicles[direction][lane]:
                    if vehicle.vehicleClass == 'ambulance' and vehicle.crossed == 0:
                        active_ambulances += 1
        
        amb_count = font.render(f"Active Ambulances: {active_ambulances}", True, (255, 255, 255))
        screen.blit(amb_count, (screenWidth - 300, 50))
         
         # Prominently Display Detection Information
        detection_box = pygame.Surface((300, 250), pygame.SRCALPHA)
        # detection_box.fill((0, 0, 0, 128))  # Semi-transparent black background
        detection_box.fill((0, 0, 0, 255))  # Completely opaque black background
        screen.blit(detection_box, (10, 10))

        # Detection Information Rendering
        detection_texts = [
            f"Lane Detection: {detection_info['lane'] + 1 if detection_info['lane'] is not None else 'N/A'}",
            "Vehicle Counts:",
            f"Cars: {detection_info['vehicle_counts']['cars']}",
            f"Bikes: {detection_info['vehicle_counts']['bikes']}",
            f"Buses: {detection_info['vehicle_counts']['buses']}",
            f"Trucks: {detection_info['vehicle_counts']['trucks']}",
            f"Rickshaws: {detection_info['vehicle_counts']['rickshaws']}",
            f"Allocated Green Time: {detection_info['allocated_time']} sec",
            f"Last Detection: {time.strftime('%H:%M:%S', time.localtime(detection_info['last_detection_time']))}"
        ]

        # Render detection texts
        for i, text in enumerate(detection_texts):
            text_surface = font.render(text, True, (255, 255, 255))  # White text
            screen.blit(text_surface, (20, 20 + i*30))
            
        # Display ambulance priority mode if active
        if isAmbulancePriority:
            priority_box = pygame.Surface((400, 80), pygame.SRCALPHA)
            priority_box.fill((255, 0, 0, 150))  # Semi-transparent red for emergency
            screen.blit(priority_box, (screenWidth/2 - 200, 10))
            
            priority_text = font.render("!!! AMBULANCE PRIORITY MODE !!!", True, (255, 255, 255))
            screen.blit(priority_text, (screenWidth/2 - 180, 30))
            
            if interruptedLane is not None:
                interrupt_text = font.render(f"Lane {interruptedLane+1} interrupted", True, (255, 255, 255))
                screen.blit(interrupt_text, (screenWidth/2 - 140, 60))

        # Rest of the existing rendering code remains the same...
        for i in range(0,noOfSignals):  # display signal and set timer according to current status: green, yello, or red
            if(i==currentGreen):
                if(currentYellow==1):
                    if(signals[i].yellow==0):
                        signals[i].signalText = "STOP"
                    else:
                        signals[i].signalText = signals[i].yellow
                    screen.blit(yellowSignal, signalCoods[i])
                else:
                    if(signals[i].green==0):
                        signals[i].signalText = "SLOW"
                    else:
                        signals[i].signalText = signals[i].green
                    screen.blit(greenSignal, signalCoods[i])
            else:
                if(signals[i].red<=10):
                    if(signals[i].red==0):
                        signals[i].signalText = "GO"
                    else:
                        signals[i].signalText = signals[i].red
                else:
                    signals[i].signalText = "---"
                screen.blit(redSignal, signalCoods[i])
        signalTexts = ["","","",""]

        # display signal timer and vehicle count
        for i in range(0,noOfSignals):  
            signalTexts[i] = font.render(str(signals[i].signalText), True, white, black)
            screen.blit(signalTexts[i], signalTimerCoods[i]) 
    
            # Display current waiting vehicle label
            screen.blit(countLabel[i], waitingLabelCoods[i])
    
            # Display current waiting vehicle count
            displayText = getCurrentVehicleCount(directionNumbers[i])
            waitingCountText = font.render(str(displayText), True, black, white)
            screen.blit(waitingCountText, waitingCountCoods[i])
    
            # Display crossed vehicles count
            # displayCrossed = vehicles[directionNumbers[i]]['crossed']
            # vehicleCountTexts[i] = font.render(str(displayCrossed), True, black, white)
            # screen.blit(vehicleCountTexts[i], vehicleCountCoods[i])

        timeElapsedText = font.render(("Time Elapsed: "+str(timeElapsed)), True, black, white)
        screen.blit(timeElapsedText,(1100,50))

        # display the vehicles
        for vehicle in simulation:  
            screen.blit(vehicle.currentImage, [vehicle.x, vehicle.y])
            vehicle.move()
        pygame.display.update()

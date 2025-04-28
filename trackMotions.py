import serial
import time
import statistics #used to get standard deviation
from VIP.logData import store_data_csv


# === CONFIG ===
COM_PORT = 'COM3'
BAUD_RATE = 1000000
OUTPUT_FILE = 'arduino_log0.csv'

# === SETUP ===
ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # Let Arduino reset

movements = [] #gets list of movements
# change = 20
samplingRate = 50 #how many samples to check for motion start/stop
variance = 0.3 #min standard deviation for checking if motion start/stop
rawData = [] #raw data collected
motion = [] #cleaned data for each motion
minMotions = samplingRate #min length of a motion
isMotion = False #checking if motion is happening or not
count = 0

# with open(OUTPUT_FILE, 'w', newline='') as f:
    # f.write("timestamp,data\n")  # CSV header
print("Logging started. Press Ctrl+C to stop.\n")

try:
    while True:

        if ser.in_waiting > 0:
            line = ser.readline().decode(errors='ignore').strip()
            # time.sleep(0.001)  #delay
            if line[1:].isnumeric() or line.isnumeric():
                count+=1
                if count % (samplingRate//1) != 0:
                    continue
                data = int(line) #scale value to larger number
                rawData.append(data) #append data to rawData
                if len(rawData) % samplingRate == 0 and len(rawData): #if reached new sample checkpoint
                    past = rawData[-samplingRate:] #get new sample
                    stdev = statistics.stdev(past) #get standard devation of sample

                    if stdev > variance: #if sample variance is greater than threshold
                        isMotion = True #start motion
                        motion += past #add motion sample data to our motion
                    elif isMotion and stdev <= variance and len(motion) > minMotions: #if motion was happeneing and motion stopped and big enough motion
                        isMotion = False #stop motion
                        movements.append(motion[:-samplingRate]) #append motion to list of movements
                        store_data_csv(movements, "encoder.csv")
                        motion = [] #reset motion
                
                    print(movements)
            # timestamp = time.time()
            # f.write(f"{timestamp},{line}\n")
            # f.flush()  # optional, ensures data is saved instantly
            # print(f"{timestamp}: {line}")
except KeyboardInterrupt:
    print("\nLogging stopped.")
finally:                                     
    ser.close()

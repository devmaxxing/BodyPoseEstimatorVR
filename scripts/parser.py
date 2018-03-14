import json

class Parser:
    def __init__(self):
        self.data = ""

    def Parse(self, fileName): #returns list of length = frame count. each elem is list of length 21 (no spine)
        file = open(fileName)
        self.data = json.load(file)
        file.close()
        result = list()
        currentFrame = list()
        
        self.data = self.data["data"]

        for i in range (len(self.data)):
            currentFrame = []
            currentFrame.append(self.data[i]["Head"]["_x"])
            currentFrame.append(self.data[i]["Head"]["_y"])
            currentFrame.append(self.data[i]["Head"]["_z"])
            currentFrame.append(self.data[i]["Head"]["_w"])
            currentFrame.append(self.data[i]["Head"]["posX"])
            currentFrame.append(self.data[i]["Head"]["posY"])
            currentFrame.append(self.data[i]["Head"]["posZ"])
            
            currentFrame.append(self.data[i]["LeftHand"]["_x"])
            currentFrame.append(self.data[i]["LeftHand"]["_y"])
            currentFrame.append(self.data[i]["LeftHand"]["_z"])
            currentFrame.append(self.data[i]["LeftHand"]["_w"])
            currentFrame.append(self.data[i]["LeftHand"]["posX"])
            currentFrame.append(self.data[i]["LeftHand"]["posY"])
            currentFrame.append(self.data[i]["LeftHand"]["posZ"])

            currentFrame.append(self.data[i]["RightHand"]["_x"])
            currentFrame.append(self.data[i]["RightHand"]["_y"])
            currentFrame.append(self.data[i]["RightHand"]["_z"])
            currentFrame.append(self.data[i]["RightHand"]["_w"])
            currentFrame.append(self.data[i]["RightHand"]["posX"])
            currentFrame.append(self.data[i]["RightHand"]["posY"])
            currentFrame.append(self.data[i]["RightHand"]["posZ"])


            result.append(currentFrame)

        return result

    def ParseSpine(self, fileName): #returns list of length = frame count. each elem is list of length 7 (spine only)
        file = open(fileName)
        self.data = json.load(file)
        file.close()
        result = list()
        currentFrame = list()
        
        self.data = self.data["data"]

        for i in range (len(self.data)):
            currentFrame = []
            currentFrame.append(self.data[i]["Spine"]["_x"])
            currentFrame.append(self.data[i]["Spine"]["_y"])
            currentFrame.append(self.data[i]["Spine"]["_z"])
            currentFrame.append(self.data[i]["Spine"]["_w"])
            currentFrame.append(self.data[i]["Spine"]["posX"])
            currentFrame.append(self.data[i]["Spine"]["posY"])
            currentFrame.append(self.data[i]["Spine"]["posZ"])

            result.append(currentFrame)

        return result

    def ParseWithSpine(self, fileName): #returns list of length = frame count. each elem is a list of length 28 (all input including spine of previous frame)
        file = open(fileName)
        self.data = json.load(file)
        file.close()
        result = list()
        currentFrame = list()
        
        self.data = self.data["data"]
        
        for i in range (1, len(self.data)):
            currentFrame = []
            
            currentFrame.append(self.data[i]["Head"]["_x"])
            currentFrame.append(self.data[i]["Head"]["_y"])
            currentFrame.append(self.data[i]["Head"]["_z"])
            currentFrame.append(self.data[i]["Head"]["_w"])
            currentFrame.append(self.data[i]["Head"]["posX"])
            currentFrame.append(self.data[i]["Head"]["posY"])
            currentFrame.append(self.data[i]["Head"]["posZ"])
            
            currentFrame.append(self.data[i]["LeftHand"]["_x"])
            currentFrame.append(self.data[i]["LeftHand"]["_y"])
            currentFrame.append(self.data[i]["LeftHand"]["_z"])
            currentFrame.append(self.data[i]["LeftHand"]["_w"])
            currentFrame.append(self.data[i]["LeftHand"]["posX"])
            currentFrame.append(self.data[i]["LeftHand"]["posY"])
            currentFrame.append(self.data[i]["LeftHand"]["posZ"])

            currentFrame.append(self.data[i]["RightHand"]["_x"])
            currentFrame.append(self.data[i]["RightHand"]["_y"])
            currentFrame.append(self.data[i]["RightHand"]["_z"])
            currentFrame.append(self.data[i]["RightHand"]["_w"])
            currentFrame.append(self.data[i]["RightHand"]["posX"])
            currentFrame.append(self.data[i]["RightHand"]["posY"])
            currentFrame.append(self.data[i]["RightHand"]["posZ"])

            currentFrame.append(self.data[i-1]["Spine"]["_x"])
            currentFrame.append(self.data[i-1]["Spine"]["_y"])
            currentFrame.append(self.data[i-1]["Spine"]["_z"])
            currentFrame.append(self.data[i-1]["Spine"]["_w"])
            currentFrame.append(self.data[i-1]["Spine"]["posX"])
            currentFrame.append(self.data[i-1]["Spine"]["posY"])
            currentFrame.append(self.data[i-1]["Spine"]["posZ"])


            result.append(currentFrame)

        return result
